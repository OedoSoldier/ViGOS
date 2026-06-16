"""Trainer for Visual Causal On-Policy Self-Distillation."""

from __future__ import annotations

import inspect
import json
import os
from contextlib import contextmanager, nullcontext
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from accelerate.utils import is_peft_model
from transformers import Trainer, TrainerCallback

from .answer_utils import extract_boxed_content, normalize_reference_answer
from .data_collator import THINK_PREFILL
from .losses import masked_kl_loss

try:
    from mathruler.grader import grade_answer
except ImportError:  # pragma: no cover - mathruler is installed in the project env.
    grade_answer = None

try:
    from trl.models.utils import unwrap_model_for_generation
except Exception:  # pragma: no cover - fallback for older TRL builds.

    @contextmanager
    def unwrap_model_for_generation(model: nn.Module, accelerator: Any):
        yield accelerator.unwrap_model(model)


MODEL_INPUT_KEYS = {
    "input_ids",
    "attention_mask",
    "pixel_values",
    "image_grid_thw",
    "pixel_values_videos",
    "video_grid_thw",
    "second_per_grid_ts",
}

class ViGOSVLLMSyncCallback(TrainerCallback):
    """Keep the vLLM rollout policy aligned with trainable weights."""

    def __init__(self, trainer: "ViGOSTrainer"):
        self.trainer = trainer

    def on_train_begin(self, args, state, control, **kwargs):
        if not self.trainer.use_vllm:
            return control
        self.trainer._move_model_to_vllm()
        self.trainer._last_vllm_sync_step = state.global_step
        return control

    def on_step_end(self, args, state, control, **kwargs):
        if not self.trainer.use_vllm:
            return control
        if not getattr(self.trainer.accelerator, "sync_gradients", True):
            return control
        if state.max_steps > 0 and state.global_step >= state.max_steps:
            return control
        if state.global_step == self.trainer._last_vllm_sync_step:
            return control
        if state.global_step % self.trainer.vllm_sync_frequency != 0:
            return control
        self.trainer._move_model_to_vllm()
        self.trainer._last_vllm_sync_step = state.global_step
        return control


class ViGOSTrainer(Trainer):
    def __init__(
        self,
        *args: Any,
        model_name_or_path: str,
        processor: Any,
        max_prompt_length: int = 32768,
        max_completion_length: int = 128,
        generation_temperature: float = 0.7,
        generation_top_p: float = 0.8,
        generation_top_k: int = 20,
        distill_temperature: float = 1.0,
        lambda_perception: float = 0.1,
        lambda_reasoning: float = 0.5,
        lambda_ref: float = 2.0,
        token_loss_clip: float | None = None,
        description_last_token_clip: float | None = 0.05,
        reasoning_first_token_clip: float | None = 0.05,
        presence_penalty: float = 0.0,
        repetition_penalty: float = 1.0,
        min_p: float = 0.0,
        use_vllm: bool = False,
        vllm_mode: str = "colocate",
        vllm_gpu_memory_utilization: float = 0.6,
        vllm_tensor_parallel_size: int = 1,
        vllm_sync_frequency: int = 1,
        vllm_max_model_len: int | None = None,
        vllm_max_num_seqs: int | None = None,
        vllm_disable_custom_all_reduce: bool = False,
        vllm_server_base_url: str | None = None,
        vllm_server_host: str = "127.0.0.1",
        vllm_server_port: int = 8000,
        vllm_server_timeout: float = 300.0,
        vllm_server_group_port: int = 51216,
        vllm_server_request_batch_size: int | None = None,
        completion_log_steps: int = 2,
        completion_log_max_samples: int = 16,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.model_name_or_path = model_name_or_path
        self.processor = processor
        self.max_prompt_length = max_prompt_length
        self.max_completion_length = max_completion_length
        self.generation_temperature = generation_temperature
        self.generation_top_p = generation_top_p
        self.generation_top_k = generation_top_k
        self.distill_temperature = distill_temperature
        self.lambda_perception = lambda_perception
        self.lambda_reasoning = lambda_reasoning
        self.lambda_ref = lambda_ref
        self.token_loss_clip = token_loss_clip
        self.description_last_token_clip = description_last_token_clip
        self.reasoning_first_token_clip = reasoning_first_token_clip
        self.fixed_teacher = True
        self.presence_penalty = presence_penalty
        self.repetition_penalty = repetition_penalty
        self.min_p = min_p
        self.use_vllm = use_vllm
        self.vllm_mode = vllm_mode
        self.vllm_gpu_memory_utilization = vllm_gpu_memory_utilization
        self.vllm_tensor_parallel_size = vllm_tensor_parallel_size
        self.vllm_sync_frequency = max(1, vllm_sync_frequency)
        self.vllm_max_model_len = vllm_max_model_len
        self.vllm_max_num_seqs = vllm_max_num_seqs
        self.vllm_disable_custom_all_reduce = vllm_disable_custom_all_reduce
        self.vllm_server_base_url = vllm_server_base_url
        self.vllm_server_host = vllm_server_host
        self.vllm_server_port = vllm_server_port
        self.vllm_server_timeout = vllm_server_timeout
        self.vllm_server_group_port = vllm_server_group_port
        self.vllm_server_request_batch_size = (
            max(1, int(vllm_server_request_batch_size))
            if vllm_server_request_batch_size
            else None
        )
        self.completion_log_steps = max(0, int(completion_log_steps))
        self.completion_log_max_samples = max(1, int(completion_log_max_samples))
        self._last_vllm_sync_step = -1
        self._last_completion_log_step = -1
        self._warned_wandb_completion_snapshot_failure = False
        self._warned_unsupported_generation_args: set[str] = set()
        self._text_token_cache: dict[str, list[int]] = {}
        self._last_loss_metrics: dict[str, float] = {}
        self._loss_metric_sums: dict[str, float] = {}
        self._loss_metric_counts: dict[str, float] = {}

        if self.use_vllm:
            # In colocate mode each training process owns a local vLLM engine. The
            # sync callback pushes freshly updated train weights into that engine.
            self._init_vllm()
            self.add_callback(ViGOSVLLMSyncCallback(self))

    def compute_loss(
        self,
        model: nn.Module,
        inputs: dict[str, Any],
        return_outputs: bool = False,
        num_items_in_batch: int | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, Any]:
        # ViGOS scores one student rollout with teacher contexts: image-only
        # perception for the description span, privileged reasoning for the
        # think/answer span, and fixed Reference KL on fallback rows.
        rollout_batch = self._build_vigos_rollout_batch(model, inputs)
        inputs = rollout_batch["raw_inputs"]
        student_prompt = rollout_batch["student_prompt"]
        perception_prompt = rollout_batch["perception_prompt"]
        reasoning_prompt = rollout_batch["reasoning_prompt"]
        reference_prompt = rollout_batch["reference_prompt"]
        rollout = rollout_batch["rollout"]
        self._maybe_log_completion_snapshot(inputs, rollout)
        generated_ids = rollout["generated_ids"]
        generated_attention_mask = rollout["generated_attention_mask"]
        completion_ids = rollout["completion_ids"]
        completion_attention = rollout["completion_attention_mask"].to(dtype=torch.bool)
        description_mask = (
            rollout["description_loss_mask"].to(dtype=torch.bool) & completion_attention
        )
        reasoning_token_ids = rollout["reasoning_token_ids"]
        reasoning_positions = rollout["reasoning_positions"]
        reasoning_attention = rollout["reasoning_attention_mask"].to(dtype=torch.bool)
        full_completion_fallback_mask = rollout["full_completion_fallback_mask"].to(
            dtype=torch.bool
        )
        non_fallback_rows = ~full_completion_fallback_mask
        description_post_clip_mask = self._last_active_token_mask(
            description_mask & non_fallback_rows.unsqueeze(1)
        )
        reasoning_post_clip_mask = self._first_active_token_mask(
            reasoning_attention & non_fallback_rows.unsqueeze(1)
        )
        ref_mask = full_completion_fallback_mask.unsqueeze(1) & completion_attention
        student_inputs = self._with_completion(
            student_prompt,
            full_input_ids=generated_ids,
            full_attention_mask=generated_attention_mask,
        )
        perception_inputs = self._append_completion(
            perception_prompt,
            completion_ids,
            rollout["completion_attention_mask"],
        )
        reasoning_inputs = self._append_completion(
            reasoning_prompt,
            reasoning_token_ids,
            rollout["reasoning_token_attention_mask"],
        )

        student_inputs["logits_to_keep"] = completion_ids.shape[1] + 1

        lambda_ref = float(getattr(self, "lambda_ref", 0.0) or 0.0)
        description_last_token_clip = self._positive_clip_or_none(
            getattr(self, "description_last_token_clip", 0.05)
        )
        reasoning_first_token_clip = self._positive_clip_or_none(
            getattr(self, "reasoning_first_token_clip", 0.05)
        )
        ref_logits = None
        compute_ref = lambda_ref > 0.0
        ref_inputs = None
        if compute_ref:
            ref_inputs = self._append_completion(
                reference_prompt,
                completion_ids,
                rollout["completion_attention_mask"],
            )

        teacher_jobs = [
            {
                "name": "perception",
                "inputs": perception_inputs,
                "completion_length": completion_ids.shape[1],
            },
            {
                "name": "reasoning",
                "inputs": reasoning_inputs,
                "completion_length": reasoning_token_ids.shape[1],
            },
        ]
        if compute_ref:
            if ref_inputs is None:
                raise RuntimeError("Reference inputs were not built.")
            teacher_jobs.append(
                {
                    "name": "ref",
                    "inputs": ref_inputs,
                    "completion_length": completion_ids.shape[1],
                }
            )
        teacher_logits = self._batched_teacher_completion_logits(model, teacher_jobs)
        perception_logits = teacher_logits["perception"]
        reasoning_logits = teacher_logits["reasoning"]
        if compute_ref:
            ref_logits = teacher_logits["ref"]
        del teacher_logits

        student_outputs = model(**student_inputs)
        student_logits = self._completion_logits(
            student_outputs.logits, completion_ids.shape[1]
        )
        del student_outputs

        student_reasoning_logits = self._gather_token_logits(
            student_logits,
            reasoning_positions,
        )
        perception_loss = masked_kl_loss(
            perception_logits,
            student_logits,
            description_mask,
            temperature=self.distill_temperature,
            token_clip=self.token_loss_clip,
            post_clip_mask=description_post_clip_mask,
            post_token_clip=description_last_token_clip,
        )
        perception_loss, _, perception_loss_numerator, perception_loss_count = (
            self._distributed_masked_loss_with_stats(
                perception_loss,
                description_mask,
            )
        )
        reasoning_loss = masked_kl_loss(
            reasoning_logits,
            student_reasoning_logits,
            reasoning_attention,
            temperature=self.distill_temperature,
            token_clip=self.token_loss_clip,
            post_clip_mask=reasoning_post_clip_mask,
            post_token_clip=reasoning_first_token_clip,
        )
        reasoning_loss, _, reasoning_loss_numerator, reasoning_loss_count = (
            self._distributed_masked_loss_with_stats(
                reasoning_loss,
                reasoning_attention,
            )
        )
        del perception_logits, reasoning_logits
        if ref_logits is not None:
            loss_ref = masked_kl_loss(
                student_logits,
                ref_logits,
                ref_mask,
                temperature=self.distill_temperature,
                token_clip=self.token_loss_clip,
            )
        else:
            loss_ref = student_logits.sum() * 0.0
        loss_ref, _, ref_loss_numerator, ref_loss_count = (
            self._distributed_masked_loss_with_stats(
                loss_ref,
                ref_mask,
            )
        )
        if ref_logits is not None:
            del ref_logits

        loss = (
            self.lambda_perception * perception_loss
            + self.lambda_reasoning * reasoning_loss
            + lambda_ref * loss_ref
        )

        rollout_answer_correct = self._rollout_answer_correctness(inputs, rollout)
        _, answer_correct_count, answer_count = self._distributed_rate_stats(
            rollout_answer_correct
        )
        _, fallback_count, fallback_total = self._distributed_rate_stats(
            rollout["full_completion_fallback_mask"]
        )
        _, description_available_count, description_available_total = (
            self._distributed_rate_stats(rollout["description_available_mask"])
        )
        _, reasoning_available_count, reasoning_available_total = (
            self._distributed_rate_stats(rollout["reasoning_available_mask"])
        )
        _, tagless_count, tagless_total = self._distributed_rate_stats(
            rollout["tagless_completion_mask"]
        )
        _, description_token_count, description_token_total = self._distributed_rate_stats(
            description_mask
        )
        _, reasoning_token_count, reasoning_token_total = self._distributed_rate_stats(
            reasoning_attention
        )
        self._record_loss_metrics(
            {
                "loss_perception": (perception_loss_numerator, perception_loss_count),
                "loss_reasoning": (reasoning_loss_numerator, reasoning_loss_count),
                "loss_ref": (ref_loss_numerator, ref_loss_count),
                "answer_accuracy": (answer_correct_count, answer_count),
                "full_completion_fallback_rate": (fallback_count, fallback_total),
                "description_supervision_rate": (
                    description_available_count,
                    description_available_total,
                ),
                "reasoning_supervision_rate": (
                    reasoning_available_count,
                    reasoning_available_total,
                ),
                "tagless_rollout_rate": (tagless_count, tagless_total),
                "description_token_ratio": (
                    description_token_count,
                    description_token_total,
                ),
                "reasoning_token_ratio": (
                    reasoning_token_count,
                    reasoning_token_total,
                ),
            }
        )

        if return_outputs:
            return loss, {"logits": student_logits.detach()}
        return loss

    def log(self, logs: dict[str, float], *args: Any, **kwargs: Any) -> None:
        if self._last_loss_metrics:
            logs = {**logs, **self._last_loss_metrics}
        result = super().log(logs, *args, **kwargs)
        self._reset_loss_metric_accumulators()
        return result

    @staticmethod
    def _positive_clip_or_none(value: float | None) -> float | None:
        if value is None:
            return None
        value = float(value)
        if value <= 0:
            return None
        return value

    def _maybe_log_completion_snapshot(
        self,
        raw_inputs: dict[str, Any],
        rollout: dict[str, Any],
    ) -> None:
        interval = int(getattr(self, "completion_log_steps", 0) or 0)
        if interval <= 0:
            return
        state = getattr(self, "state", None)
        step = int(getattr(state, "global_step", 0) or 0)
        if step % interval != 0:
            return
        if step == int(getattr(self, "_last_completion_log_step", -1)):
            return
        self._last_completion_log_step = step

        local_records = self._completion_snapshot_records(raw_inputs, rollout, step)
        gathered_records: list[list[dict[str, Any]] | None]
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            gathered_records = [
                None for _ in range(torch.distributed.get_world_size())
            ]
            torch.distributed.all_gather_object(gathered_records, local_records)
        else:
            gathered_records = [local_records]

        accelerator = getattr(self, "accelerator", None)
        if accelerator is not None and not getattr(accelerator, "is_main_process", True):
            return

        records = [
            record
            for rank_records in gathered_records
            for record in (rank_records or [])
        ]
        records = records[: int(getattr(self, "completion_log_max_samples", 16))]
        if not records:
            return

        output_dir = Path(str(getattr(self.args, "output_dir", "runs/vigos")))
        sample_dir = output_dir / "completion_samples"
        sample_dir.mkdir(parents=True, exist_ok=True)
        snapshot_path = sample_dir / f"completions_step{step:06d}.jsonl"
        with snapshot_path.open("w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")

        self._log_completion_snapshot_to_wandb(records, snapshot_path, step)

    def _completion_snapshot_records(
        self,
        raw_inputs: dict[str, Any],
        rollout: dict[str, Any],
        step: int,
    ) -> list[dict[str, Any]]:
        completion_ids = rollout["completion_ids"]
        completion_attention = rollout["completion_attention_mask"]
        batch_size = completion_ids.shape[0]
        accelerator = getattr(self, "accelerator", None)
        rank = int(getattr(accelerator, "process_index", 0) or 0)
        epoch = getattr(getattr(self, "state", None), "epoch", None)
        sample_ids = self._metadata_values(raw_inputs.get("sample_ids"), batch_size)
        problems = self._metadata_values(raw_inputs.get("vigos_problems"), batch_size)
        references = self._metadata_values(
            raw_inputs.get("vigos_references"), batch_size
        )
        answers = self._metadata_values(raw_inputs.get("vigos_answers"), batch_size)
        description_texts = rollout.get("description_texts") or [""] * batch_size

        records = []
        for row_idx in range(batch_size):
            valid_length = int(completion_attention[row_idx].sum().item())
            completion_text = self._decode_token_ids(
                completion_ids[row_idx, :valid_length],
                skip_special_tokens=True,
            )
            answer_correct = self._answers_match(
                extract_boxed_content(completion_text),
                answers[row_idx],
            )
            records.append(
                {
                    "global_step": step,
                    "epoch": epoch,
                    "rank": rank,
                    "local_row": row_idx,
                    "sample_id": sample_ids[row_idx],
                    "problem": problems[row_idx],
                    "reference": references[row_idx],
                    "completion": completion_text,
                    "answer_correct": answer_correct,
                    "description_text": description_texts[row_idx],
                    "has_think": bool(rollout["valid_mask"][row_idx].item()),
                    "description_available": bool(
                        rollout["description_available_mask"][row_idx].item()
                    ),
                    "reasoning_available": bool(
                        rollout["reasoning_available_mask"][row_idx].item()
                    ),
                }
            )
        return records

    @staticmethod
    def _metadata_values(values: Any, batch_size: int) -> list[Any]:
        if values is None:
            return [None] * batch_size
        if isinstance(values, torch.Tensor):
            flat = values.detach().cpu().flatten().tolist()
            return [flat[idx] if idx < len(flat) else None for idx in range(batch_size)]
        if isinstance(values, (list, tuple)):
            return [values[idx] if idx < len(values) else None for idx in range(batch_size)]
        return [values] * batch_size

    def _log_completion_snapshot_to_wandb(
        self,
        records: list[dict[str, Any]],
        snapshot_path: Path,
        step: int,
    ) -> None:
        try:
            import wandb
        except ImportError:
            return
        if wandb.run is None:
            return
        columns = [
            "global_step",
            "epoch",
            "rank",
            "local_row",
            "sample_id",
            "problem",
            "reference",
            "completion",
            "answer_correct",
            "description_text",
            "has_think",
            "description_available",
            "reasoning_available",
        ]
        try:
            table = wandb.Table(columns=columns)
            for record in records:
                table.add_data(*(record.get(column) for column in columns))
            wandb.log(
                {
                    "rollout/completion_samples": table,
                    "rollout/completion_sample_count": len(records),
                },
                step=step,
            )
        except Exception as exc:
            if not getattr(self, "_warned_wandb_completion_snapshot_failure", False):
                print(
                    "Warning: failed to sync completion snapshot to W&B; "
                    f"continuing with local JSONL only. {type(exc).__name__}: {exc}",
                    flush=True,
                )
                self._warned_wandb_completion_snapshot_failure = True
            return
        try:
            wandb.save(str(snapshot_path), policy="now")
        except Exception as exc:
            if not getattr(self, "_warned_wandb_completion_snapshot_failure", False):
                print(
                    "Warning: failed to save completion snapshot artifact to W&B; "
                    f"continuing with local JSONL only. {type(exc).__name__}: {exc}",
                    flush=True,
                )
                self._warned_wandb_completion_snapshot_failure = True

    def _distributed_rate(self, values: torch.Tensor) -> float:
        rate, _, _ = self._distributed_rate_stats(values)
        return rate

    def _distributed_rate_stats(self, values: torch.Tensor) -> tuple[float, float, float]:
        values = values.detach().to(dtype=torch.float32)
        local_stats = torch.stack(
            [
                values.sum(),
                values.new_tensor(float(values.numel()), dtype=torch.float32),
            ]
        )
        accelerator = getattr(self, "accelerator", None)
        if accelerator is not None:
            try:
                gathered = accelerator.gather_for_metrics(local_stats)
                local_stats = gathered.reshape(-1, 2).sum(dim=0)
            except Exception:
                pass
        denominator = float(local_stats[1].detach().cpu())
        if denominator <= 0.0:
            return 0.0, float(local_stats[0].detach().cpu()), denominator
        numerator = float(local_stats[0].detach().cpu())
        return float((local_stats[0] / local_stats[1]).detach().cpu()), numerator, denominator

    def _distributed_masked_loss(
        self,
        local_loss: torch.Tensor,
        token_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, float]:
        objective_loss, logged_loss, _, _ = self._distributed_masked_loss_with_stats(
            local_loss,
            token_mask,
        )
        return objective_loss, logged_loss

    def _distributed_masked_loss_with_stats(
        self,
        local_loss: torch.Tensor,
        token_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, float, float, float]:
        """Return a globally normalized objective loss and matching log value.

        ``masked_kl_loss`` returns a local active-token mean. DDP then averages
        gradients across ranks, which would underweight sparse masks when only a
        subset of ranks has active tokens. Convert the local mean back to a
        numerator and normalize by the global active-token count while
        compensating for DDP's gradient averaging.
        """
        local_count = (
            token_mask.detach()
            .to(device=local_loss.device, dtype=torch.float32)
            .sum()
        )
        local_numerator = local_loss.to(dtype=torch.float32) * local_count
        global_count = local_count.detach().clone()
        global_numerator = local_numerator.detach().to(dtype=torch.float32)
        world_size = 1
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.all_reduce(
                global_count,
                op=torch.distributed.ReduceOp.SUM,
            )
            torch.distributed.all_reduce(
                global_numerator,
                op=torch.distributed.ReduceOp.SUM,
            )
            world_size = torch.distributed.get_world_size()

        denominator = global_count.clamp_min(1.0)
        gradient_loss = local_numerator * (float(world_size) / denominator)
        logged_loss = global_numerator / denominator
        value_loss = logged_loss.to(device=local_loss.device, dtype=torch.float32)
        objective_loss = gradient_loss + (value_loss - gradient_loss.detach())
        return (
            objective_loss,
            float(logged_loss.detach().cpu()),
            float(global_numerator.detach().cpu()),
            float(global_count.detach().cpu()),
        )

    def _record_loss_metrics(
        self,
        metric_stats: dict[str, tuple[float, float]],
    ) -> dict[str, float]:
        """Accumulate numerator/count metrics across micro-batches until log()."""
        sums = getattr(self, "_loss_metric_sums", None)
        counts = getattr(self, "_loss_metric_counts", None)
        if sums is None or counts is None:
            sums = {}
            counts = {}
            self._loss_metric_sums = sums
            self._loss_metric_counts = counts

        for name, (numerator, denominator) in metric_stats.items():
            sums[name] = sums.get(name, 0.0) + float(numerator)
            counts[name] = counts.get(name, 0.0) + float(denominator)

        self._last_loss_metrics = self._loss_metric_snapshot()
        return self._last_loss_metrics

    def _loss_metric_snapshot(self) -> dict[str, float]:
        sums = getattr(self, "_loss_metric_sums", {})
        counts = getattr(self, "_loss_metric_counts", {})
        metrics: dict[str, float] = {}
        for name, numerator in sums.items():
            denominator = float(counts.get(name, 0.0))
            metrics[name] = float(numerator) / denominator if denominator > 0.0 else 0.0
        return metrics

    def _reset_loss_metric_accumulators(self) -> None:
        self._loss_metric_sums = {}
        self._loss_metric_counts = {}
        self._last_loss_metrics = {}

    @staticmethod
    def _first_active_token_mask(token_mask: torch.Tensor) -> torch.Tensor:
        token_mask = token_mask.to(dtype=torch.bool)
        if token_mask.ndim != 2:
            raise ValueError(
                "token_mask must be a 2D [batch, tokens] tensor, got "
                f"{tuple(token_mask.shape)}."
            )
        if token_mask.shape[1] == 0:
            return torch.zeros_like(token_mask, dtype=torch.bool)
        token_positions = torch.arange(
            token_mask.shape[1],
            device=token_mask.device,
        ).unsqueeze(0)
        first_positions = torch.where(
            token_mask,
            token_positions,
            token_mask.new_full(token_mask.shape, token_mask.shape[1], dtype=torch.long),
        ).min(dim=1).values
        active_rows = first_positions < token_mask.shape[1]
        boundary_mask = torch.zeros_like(token_mask, dtype=torch.bool)
        if active_rows.any():
            row_indices = torch.arange(token_mask.shape[0], device=token_mask.device)
            boundary_mask[row_indices[active_rows], first_positions[active_rows]] = True
        return boundary_mask

    @staticmethod
    def _last_active_token_mask(token_mask: torch.Tensor) -> torch.Tensor:
        token_mask = token_mask.to(dtype=torch.bool)
        if token_mask.ndim != 2:
            raise ValueError(
                "token_mask must be a 2D [batch, tokens] tensor, got "
                f"{tuple(token_mask.shape)}."
            )
        if token_mask.shape[1] == 0:
            return torch.zeros_like(token_mask, dtype=torch.bool)
        token_positions = torch.arange(
            token_mask.shape[1],
            device=token_mask.device,
        ).unsqueeze(0)
        last_positions = torch.where(
            token_mask,
            token_positions,
            token_mask.new_full(token_mask.shape, -1, dtype=torch.long),
        ).max(dim=1).values
        active_rows = last_positions >= 0
        boundary_mask = torch.zeros_like(token_mask, dtype=torch.bool)
        if active_rows.any():
            row_indices = torch.arange(token_mask.shape[0], device=token_mask.device)
            boundary_mask[row_indices[active_rows], last_positions[active_rows]] = True
        return boundary_mask

    def _rollout_answer_correctness(
        self,
        raw_inputs: dict[str, Any],
        rollout: dict[str, Any],
    ) -> torch.Tensor:
        completion_ids = rollout["completion_ids"]
        completion_attention = rollout["completion_attention_mask"]
        batch_size = completion_ids.shape[0]
        answers = self._metadata_values(raw_inputs.get("vigos_answers"), batch_size)
        correct = torch.zeros(
            batch_size,
            dtype=torch.bool,
            device=completion_ids.device,
        )
        for row_idx in range(batch_size):
            valid_length = int(completion_attention[row_idx].sum().item())
            completion_text = self._decode_token_ids(
                completion_ids[row_idx, :valid_length],
                skip_special_tokens=True,
            )
            prediction = extract_boxed_content(completion_text)
            correct[row_idx] = self._answers_match(prediction, answers[row_idx])
        return correct

    @staticmethod
    def _answers_match(prediction: Any, reference: Any) -> bool:
        prediction_text = normalize_reference_answer(prediction)
        reference_text = normalize_reference_answer(reference)
        if not prediction_text or not reference_text:
            return False
        if grade_answer is not None:
            try:
                return bool(grade_answer(prediction_text, reference_text))
            except Exception:
                pass
        return _casefold_answer(prediction_text) == _casefold_answer(reference_text)

    def _build_vigos_rollout_batch(
        self,
        model: nn.Module,
        inputs: dict[str, Any],
    ) -> dict[str, Any]:
        student_prompt = self._prompt_inputs(inputs, "student")
        perception_prompt = self._prompt_inputs(inputs, "perception")
        reasoning_prompt = self._prompt_inputs(inputs, "reasoning")
        reference_prompt = self._prompt_inputs(inputs, "reference")
        rollout = self._generate_vigos_rollout(
            model,
            student_prompt,
            inputs,
        )
        return {
            "raw_inputs": inputs,
            "student_prompt": student_prompt,
            "perception_prompt": perception_prompt,
            "reasoning_prompt": reasoning_prompt,
            "reference_prompt": reference_prompt,
            "rollout": rollout,
            "skipped_sources": torch.tensor(
                0.0,
                dtype=torch.float32,
                device=student_prompt["input_ids"].device,
            ),
        }

    def _generate_vigos_rollout(
        self,
        model: nn.Module,
        prompt_inputs: dict[str, torch.Tensor],
        raw_inputs: dict[str, Any],
    ) -> dict[str, torch.Tensor | list[str]]:
        batch_size = prompt_inputs["input_ids"].shape[0]
        pad_token_id = self._pad_token_id()
        if pad_token_id is None:
            pad_token_id = self._eos_token_id()
        if pad_token_id is None:
            raise RuntimeError(
                "A pad or EOS token is required to pad rollout completions."
            )

        rollout = self._generate_on_policy(model, prompt_inputs, raw_inputs)
        spans = self._analyze_vigos_completion(
            rollout["completion_ids"],
            rollout["completion_attention_mask"],
        )
        completion_ids = rollout["completion_ids"]
        completion_attention = rollout["completion_attention_mask"]

        generated_ids = torch.cat([prompt_inputs["input_ids"], completion_ids], dim=1)
        generated_attention = torch.cat(
            [prompt_inputs["attention_mask"], completion_attention],
            dim=1,
        )
        masks = self._build_vigos_masks(
            completion_ids,
            completion_attention,
            spans,
        )
        return {
            "source_indices": torch.tensor(
                list(range(batch_size)),
                dtype=torch.long,
                device=prompt_inputs["input_ids"].device,
            ),
            "generated_ids": generated_ids,
            "generated_attention_mask": generated_attention,
            "completion_ids": completion_ids,
            "completion_attention_mask": completion_attention,
            "perception_completion_ids": masks["perception_completion_ids"],
            "perception_completion_attention_mask": masks[
                "perception_completion_attention_mask"
            ],
            "description_loss_mask": masks["description_loss_mask"],
            "reasoning_positions": masks["reasoning_positions"],
            "reasoning_attention_mask": masks["reasoning_attention_mask"],
            "reasoning_token_ids": masks["reasoning_token_ids"],
            "reasoning_token_attention_mask": masks["reasoning_token_attention_mask"],
            "description_texts": masks["description_texts"],
            "description_positions": masks["description_positions"],
            "description_token_ids": masks["description_token_ids"],
            "description_token_attention_mask": masks[
                "description_token_attention_mask"
            ],
            "valid_mask": masks["valid_mask"],
            "full_completion_fallback_mask": masks["full_completion_fallback_mask"],
            "description_available_mask": masks["description_available_mask"],
            "reasoning_available_mask": masks["reasoning_available_mask"],
            "tagless_completion_mask": masks["tagless_completion_mask"],
        }

    def _analyze_vigos_completion(
        self,
        completion_ids: torch.Tensor,
        completion_attention: torch.Tensor,
    ) -> list[dict[str, Any]]:
        spans = []
        for row_idx in range(completion_ids.shape[0]):
            valid_length = int(completion_attention[row_idx].sum().item())
            description_close = self._find_tag_span(
                completion_ids[row_idx],
                valid_length,
                "</description>",
            )
            think_open_any = self._find_tag_span(
                completion_ids[row_idx],
                valid_length,
                THINK_PREFILL,
            )
            think_open = None
            if description_close is not None:
                think_open = self._find_tag_span(
                    completion_ids[row_idx],
                    valid_length,
                    THINK_PREFILL,
                    start=description_close[0],
                )
            think_close = None
            think_close_any = None
            if think_open_any is not None:
                think_close_any = self._find_tag_span(
                    completion_ids[row_idx],
                    valid_length,
                    "</think>",
                    start=think_open_any[0],
                )
            boxed_answer_parts = None
            boxed_answer_parts_any = None
            if think_close_any is not None:
                boxed_answer_parts_any = self._find_boxed_answer_parts(
                    completion_ids[row_idx],
                    valid_length,
                    start=think_close_any[0],
                )
            if think_open is not None:
                think_close = self._find_tag_span(
                    completion_ids[row_idx],
                    valid_length,
                    "</think>",
                    start=think_open[0],
                )
            if think_close is not None:
                boxed_answer_parts = self._find_boxed_answer_parts(
                    completion_ids[row_idx],
                    valid_length,
                    start=think_close[0],
                )
            boxed_answer_span = (
                boxed_answer_parts.get("span")
                if boxed_answer_parts is not None
                else None
            )
            valid = (
                description_close is not None
                and think_open is not None
                and think_close is not None
                and boxed_answer_span is not None
            )
            if valid:
                description_start = 0
                description_end = description_close[0]
                reasoning_start = think_open[1]
                reasoning_end = boxed_answer_span[1]
            else:
                description_start = 0
                description_end = 0
                reasoning_start = 0
                reasoning_end = 0
            description_available = self._span_has_supervision_tokens(
                completion_ids[row_idx],
                description_start,
                description_end,
            )
            reasoning_available = self._span_has_supervision_tokens(
                completion_ids[row_idx],
                reasoning_start,
                reasoning_end,
            )
            tag_count = sum(
                tag is not None
                for tag in (description_close, think_open_any, think_close_any)
            ) + int(boxed_answer_parts_any is not None)
            spans.append(
                {
                    "valid": valid
                    and description_available
                    and reasoning_available,
                    "valid_length": valid_length,
                    "description_available": description_available,
                    "description_start": description_start,
                    "description_end": description_end,
                    "reasoning_available": reasoning_available,
                    "reasoning_start": reasoning_start,
                    "reasoning_end": reasoning_end,
                    "tag_count": tag_count,
                    "has_boxed_answer": boxed_answer_span is not None,
                }
            )
        return spans

    def _build_vigos_masks(
        self,
        completion_ids: torch.Tensor,
        completion_attention: torch.Tensor,
        spans: list[dict[str, Any]],
    ) -> dict[str, torch.Tensor | list[str]]:
        batch_size, completion_length = completion_ids.shape
        description_mask = torch.zeros(
            (batch_size, completion_length),
            dtype=torch.bool,
            device=completion_ids.device,
        )
        valid_mask = torch.zeros(
            batch_size, dtype=torch.bool, device=completion_ids.device
        )
        description_available_mask = torch.zeros(
            batch_size, dtype=torch.bool, device=completion_ids.device
        )
        reasoning_available_mask = torch.zeros(
            batch_size, dtype=torch.bool, device=completion_ids.device
        )
        tagless_completion_mask = torch.zeros(
            batch_size, dtype=torch.bool, device=completion_ids.device
        )
        full_completion_fallback_mask = torch.zeros(
            batch_size,
            dtype=torch.bool,
            device=completion_ids.device,
        )
        description_rows: list[torch.Tensor] = []
        description_position_rows: list[torch.Tensor] = []
        description_attention_rows: list[torch.Tensor] = []
        reasoning_rows: list[torch.Tensor] = []
        reasoning_position_rows: list[torch.Tensor] = []
        reasoning_attention_rows: list[torch.Tensor] = []
        description_texts: list[str] = []

        pad_token_id = self._pad_token_id()
        if pad_token_id is None:
            pad_token_id = self._eos_token_id()
        if pad_token_id is None:
            pad_token_id = 0

        for row_idx, span in enumerate(spans):
            valid = bool(span["valid"])
            valid_mask[row_idx] = valid
            description_start = int(span["description_start"])
            description_end = int(span["description_end"])
            reasoning_start = int(span["reasoning_start"])
            reasoning_end = int(span["reasoning_end"])
            fallback_full_completion = not valid
            if fallback_full_completion:
                full_completion_fallback_mask[row_idx] = True
                description_available = False
                reasoning_available = False
                description_start = description_end = 0
                reasoning_start = reasoning_end = 0
            else:
                description_available = bool(span["description_available"])
                reasoning_available = bool(span["reasoning_available"])
            tag_count = int(span["tag_count"])
            tagless_completion_mask[row_idx] = tag_count == 0

            if description_end > description_start and description_available:
                if description_available:
                    description_available_mask[row_idx] = True
                description_mask[row_idx, description_start:description_end] = True
                description_tokens = completion_ids[
                    row_idx, description_start:description_end
                ]
                description_positions = torch.arange(
                    description_start,
                    description_end,
                    dtype=torch.long,
                    device=completion_ids.device,
                )
                description_attention = torch.ones_like(
                    description_positions, dtype=torch.bool
                )
                if description_available:
                    description_texts.append(
                        self._decode_token_ids(
                            completion_ids[row_idx, description_start:description_end]
                        ).strip()
                    )
                else:
                    description_texts.append("")
            else:
                description_tokens = completion_ids.new_tensor([pad_token_id])
                description_positions = completion_ids.new_tensor([0])
                description_attention = torch.zeros(
                    (1,), dtype=torch.bool, device=completion_ids.device
                )
                description_texts.append("")
            description_rows.append(description_tokens)
            description_position_rows.append(description_positions)
            description_attention_rows.append(description_attention)

            if reasoning_end > reasoning_start and reasoning_available:
                if reasoning_available:
                    reasoning_available_mask[row_idx] = True
                reasoning_tokens = completion_ids[
                    row_idx, reasoning_start:reasoning_end
                ]
                positions = torch.arange(
                    reasoning_start,
                    reasoning_end,
                    dtype=torch.long,
                    device=completion_ids.device,
                )
                reasoning_attention = torch.ones_like(positions, dtype=torch.bool)
            else:
                reasoning_tokens = completion_ids.new_tensor([pad_token_id])
                positions = completion_ids.new_tensor([0])
                reasoning_attention = torch.zeros(
                    (1,), dtype=torch.bool, device=completion_ids.device
                )
            reasoning_rows.append(reasoning_tokens)
            reasoning_position_rows.append(positions)
            reasoning_attention_rows.append(reasoning_attention)

        description_token_ids, _ = self._pad_tensor_rows(
            description_rows,
            pad_value=pad_token_id,
        )
        description_positions, _ = self._pad_tensor_rows(
            description_position_rows,
            pad_value=0,
        )
        description_attention = torch.stack(
            [
                self._pad_1d_tensor(row, description_token_ids.shape[1], pad_value=0)
                for row in description_attention_rows
            ],
            dim=0,
        ).to(dtype=torch.bool, device=completion_ids.device)
        reasoning_token_ids, _ = self._pad_tensor_rows(
            reasoning_rows,
            pad_value=pad_token_id,
        )
        reasoning_positions, _ = self._pad_tensor_rows(
            reasoning_position_rows,
            pad_value=0,
        )
        reasoning_attention = torch.stack(
            [
                self._pad_1d_tensor(row, reasoning_token_ids.shape[1], pad_value=0)
                for row in reasoning_attention_rows
            ],
            dim=0,
        ).to(dtype=torch.bool, device=completion_ids.device)
        return {
            "perception_completion_ids": description_token_ids,
            "perception_completion_attention_mask": description_attention.to(
                dtype=completion_attention.dtype
            ),
            "description_loss_mask": description_mask
            & completion_attention.to(dtype=torch.bool),
            "description_positions": description_positions,
            "description_token_ids": description_token_ids,
            "description_token_attention_mask": description_attention.to(
                dtype=completion_attention.dtype
            ),
            "reasoning_positions": reasoning_positions,
            "reasoning_attention_mask": reasoning_attention,
            "reasoning_token_ids": reasoning_token_ids,
            "reasoning_token_attention_mask": reasoning_attention.to(
                dtype=completion_attention.dtype
            ),
            "description_texts": description_texts,
            "valid_mask": valid_mask,
            "full_completion_fallback_mask": full_completion_fallback_mask,
            "description_available_mask": description_available_mask,
            "reasoning_available_mask": reasoning_available_mask,
            "tagless_completion_mask": tagless_completion_mask,
        }

    def _find_tag_span(
        self,
        row: torch.Tensor,
        valid_length: int,
        tag: str,
        start: int = 0,
    ) -> tuple[int, int] | None:
        tag_ids = self._text_token_ids(tag)
        if not tag_ids or valid_length <= 0:
            return None
        valid = row[:valid_length]
        tag_tensor = torch.tensor(tag_ids, dtype=valid.dtype, device=valid.device)
        max_start = valid.shape[0] - tag_tensor.shape[0]
        for pos in range(max(start, 0), max_start + 1):
            if torch.equal(valid[pos : pos + tag_tensor.shape[0]], tag_tensor):
                return pos, pos + tag_tensor.shape[0]

        text = self._decode_token_ids(valid, skip_special_tokens=False)
        start_text = (
            self._decode_token_ids(valid[:start], skip_special_tokens=False)
            if start > 0
            else ""
        )
        char_pos = text.find(tag, len(start_text))
        if char_pos < 0:
            return None
        offset_span = self._token_span_from_offsets(
            valid,
            text,
            char_start=char_pos,
            char_end=char_pos + len(tag),
        )
        if offset_span is not None:
            return offset_span
        token_start = len(self._text_token_ids(text[:char_pos]))
        token_end = len(self._text_token_ids(text[: char_pos + len(tag)]))
        token_start = min(max(token_start, 0), valid_length)
        token_end = min(max(token_end, token_start), valid_length)
        return token_start, token_end

    def _find_boxed_answer_parts(
        self,
        row: torch.Tensor,
        valid_length: int,
        start: int = 0,
    ) -> dict[str, Any] | None:
        open_tag = "\\boxed{"
        open_span = self._find_tag_span(row, valid_length, open_tag, start=start)
        if open_span is None:
            return None

        valid = row[:valid_length]
        text = self._decode_token_ids(valid, skip_special_tokens=False)
        start_text = (
            self._decode_token_ids(valid[:start], skip_special_tokens=False)
            if start > 0
            else ""
        )
        char_start = text.find(open_tag, len(start_text))
        if char_start < 0:
            return {
                "span": open_span,
                "open_span": open_span,
                "close_span": None,
                "answer_text": "",
            }

        depth = 1
        char_end = None
        cursor = char_start + len(open_tag)
        while cursor < len(text):
            char = text[cursor]
            if char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    char_end = cursor + 1
                    break
            cursor += 1
        if char_end is None:
            return {
                "span": None,
                "open_span": open_span,
                "close_span": None,
                "answer_text": text[char_start + len(open_tag) :],
            }

        offset_span = self._token_span_from_offsets(
            valid,
            text,
            char_start=char_start,
            char_end=char_end,
        )
        if offset_span is None:
            token_start = len(self._text_token_ids(text[:char_start]))
            token_end = len(self._text_token_ids(text[:char_end]))
            token_start = min(max(token_start, 0), valid_length)
            token_end = min(max(token_end, token_start), valid_length)
            offset_span = (token_start, token_end)

        close_span = self._token_span_from_offsets(
            valid,
            text,
            char_start=char_end - 1,
            char_end=char_end,
        )
        if close_span is None:
            close_start = len(self._text_token_ids(text[: char_end - 1]))
            close_end = len(self._text_token_ids(text[:char_end]))
            close_start = min(max(close_start, 0), valid_length)
            close_end = min(max(close_end, close_start), valid_length)
            close_span = (close_start, close_end)

        return {
            "span": offset_span,
            "open_span": open_span,
            "close_span": close_span,
            "answer_text": text[char_start + len(open_tag) : char_end - 1],
        }

    def _find_boxed_answer_span(
        self,
        row: torch.Tensor,
        valid_length: int,
        start: int = 0,
    ) -> tuple[int, int] | None:
        parts = self._find_boxed_answer_parts(row, valid_length, start=start)
        if parts is None:
            return None
        return parts["span"]

    def _completion_matches_student_prompt_format(
        self,
        row: torch.Tensor,
        valid_length: int,
    ) -> bool:
        text = self._decode_token_ids(row[:valid_length], skip_special_tokens=True).strip()
        if not text:
            return False
        description_close = text.find("</description>")
        if description_close <= 0:
            return False
        if not text[:description_close].strip():
            return False

        after_description = description_close + len("</description>")
        think_open = text.find(THINK_PREFILL, after_description)
        if think_open < 0 or text[after_description:think_open].strip():
            return False

        after_think_open = think_open + len(THINK_PREFILL)
        think_close = text.find("</think>", after_think_open)
        if think_close < 0 or not text[after_think_open:think_close].strip():
            return False

        after_think_close = think_close + len("</think>")
        boxed_open = text.find("\\boxed{", after_think_close)
        if boxed_open < 0 or text[after_think_close:boxed_open].strip():
            return False

        boxed_close = self._find_matching_boxed_close(text, boxed_open)
        if boxed_close is None:
            return False
        answer = text[boxed_open + len("\\boxed{") : boxed_close].strip()
        if not answer:
            return False

        # The student prompt's format line is sentence-terminated. Treat an
        # optional final period as formatting punctuation, but reject extra text.
        return text[boxed_close + 1 :].strip() in {"", "."}

    @staticmethod
    def _find_matching_boxed_close(text: str, boxed_open: int) -> int | None:
        cursor = boxed_open + len("\\boxed{")
        depth = 1
        while cursor < len(text):
            char = text[cursor]
            if char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    return cursor
            cursor += 1
        return None

    def _token_span_from_offsets(
        self,
        row: torch.Tensor,
        text: str,
        *,
        char_start: int,
        char_end: int,
    ) -> tuple[int, int] | None:
        tokenizer = getattr(self.processor, "tokenizer", self.processor)
        if not getattr(tokenizer, "is_fast", False):
            return None
        try:
            encoded = tokenizer(
                text,
                add_special_tokens=False,
                return_offsets_mapping=True,
            )
        except Exception:
            return None
        input_ids = (
            encoded["input_ids"] if isinstance(encoded, dict) else encoded.input_ids
        )
        offsets = (
            encoded["offset_mapping"]
            if isinstance(encoded, dict)
            else encoded.offset_mapping
        )
        if input_ids and isinstance(input_ids[0], list):
            input_ids = input_ids[0]
            offsets = offsets[0]
        row_ids = [int(value) for value in row.detach().cpu().tolist()]
        if list(input_ids) != row_ids:
            return None

        token_start = None
        token_end = None
        for idx, (start, end) in enumerate(offsets):
            if token_start is None and end > char_start:
                token_start = idx
            if token_end is None and start >= char_end:
                token_end = idx
            if token_start is not None and token_end is not None:
                break
        if token_start is None:
            return None
        if token_end is None:
            token_end = len(offsets)
        return token_start, max(token_end, token_start)

    def _text_token_ids(self, text: str) -> list[int]:
        cached = self._text_token_cache.get(text)
        if cached is not None:
            return cached
        tokenizer = getattr(self.processor, "tokenizer", self.processor)
        if hasattr(tokenizer, "encode"):
            try:
                ids = list(tokenizer.encode(text, add_special_tokens=False))
            except TypeError:
                ids = list(tokenizer.encode(text))
        else:
            encoded = tokenizer(text, add_special_tokens=False)
            ids = (
                encoded["input_ids"] if isinstance(encoded, dict) else encoded.input_ids
            )
            if ids and isinstance(ids[0], list):
                ids = ids[0]
            ids = list(ids)
        self._text_token_cache[text] = ids
        return ids

    def _decode_token_ids(
        self,
        ids: torch.Tensor,
        *,
        skip_special_tokens: bool = True,
    ) -> str:
        tokenizer = getattr(self.processor, "tokenizer", self.processor)
        values = [int(value) for value in ids.detach().cpu().tolist()]
        if hasattr(tokenizer, "decode"):
            return tokenizer.decode(values, skip_special_tokens=skip_special_tokens)
        return "".join(str(value) for value in values)

    def _span_has_supervision_tokens(
        self,
        row: torch.Tensor,
        start: int,
        end: int,
    ) -> bool:
        start = max(0, int(start))
        end = min(max(start, int(end)), int(row.shape[0]))
        if end <= start:
            return False
        ignored = set(self._normalize_eos_token_ids(self._eos_token_id()))
        pad_token_id = self._pad_token_id()
        if pad_token_id is not None:
            ignored.add(int(pad_token_id))
        if not ignored:
            return True
        return any(int(token_id) not in ignored for token_id in row[start:end])

    @staticmethod
    def _gather_token_logits(
        logits: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        positions = positions.clamp(min=0, max=max(0, logits.shape[1] - 1))
        index = positions.unsqueeze(-1).expand(-1, -1, logits.shape[-1])
        return torch.gather(logits, dim=1, index=index)

    @staticmethod
    def _pad_tensor_rows(
        rows: list[torch.Tensor],
        pad_value: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not rows:
            raise ValueError("Cannot pad an empty row list.")
        max_length = max(max(1, int(row.numel())) for row in rows)
        padded = torch.stack(
            [
                ViGOSTrainer._pad_1d_tensor(row, max_length, pad_value=pad_value)
                for row in rows
            ],
            dim=0,
        )
        attention = torch.stack(
            [
                torch.arange(max_length, device=row.device).lt(int(row.numel()))
                for row in rows
            ],
            dim=0,
        )
        return padded, attention.to(dtype=torch.long)

    @staticmethod
    def _pad_1d_tensor(
        row: torch.Tensor,
        length: int,
        *,
        pad_value: int,
    ) -> torch.Tensor:
        if int(row.numel()) >= length:
            return row[:length]
        pad = torch.full(
            (length - int(row.numel()),),
            pad_value,
            dtype=row.dtype,
            device=row.device,
        )
        return torch.cat([row, pad], dim=0)

    def _generate_on_policy(
        self,
        model: nn.Module,
        prompt_inputs: dict[str, torch.Tensor],
        raw_inputs: dict[str, Any],
    ) -> dict[str, torch.Tensor]:
        if self.use_vllm:
            return self._generate_on_policy_vllm(raw_inputs, prompt_inputs)

        # HF generation is used as a no-gradient rollout policy when vLLM is off.
        # The sampled tokens are fed back through the train model for gradients.
        generation_kwargs: dict[str, Any] = {
            "max_new_tokens": self.max_completion_length,
            "pad_token_id": self._pad_token_id(),
            "eos_token_id": self._eos_token_id(),
            "use_cache": True,
            "repetition_penalty": self.repetition_penalty,
        }
        if self.generation_temperature > 0:
            generation_kwargs.update(
                {
                    "do_sample": True,
                    "temperature": self.generation_temperature,
                    "top_p": self.generation_top_p,
                }
            )
            if self.generation_top_k > 0:
                generation_kwargs["top_k"] = self.generation_top_k
        else:
            generation_kwargs["do_sample"] = False

        with (
            torch.no_grad(),
            unwrap_model_for_generation(model, self.accelerator) as unwrapped,
        ):
            was_training = unwrapped.training
            unwrapped.eval()
            try:
                self._maybe_add_generation_arg(
                    unwrapped,
                    generation_kwargs,
                    "presence_penalty",
                    self.presence_penalty,
                    active=self.presence_penalty != 0.0,
                )
                self._maybe_add_generation_arg(
                    unwrapped,
                    generation_kwargs,
                    "min_p",
                    self.min_p,
                    active=self.min_p > 0.0,
                )
                generated_ids = unwrapped.generate(**prompt_inputs, **generation_kwargs)
            finally:
                if was_training:
                    unwrapped.train()

        # Build completion masks from EOS/PAD so loss ignores padded tail tokens but
        # still includes the first EOS token, matching standard LM supervision.
        student_prompt_len = prompt_inputs["input_ids"].shape[1]
        completion_ids = generated_ids[:, student_prompt_len:]
        completion_attention = self._completion_attention_from_token_ids(
            completion_ids,
            pad_token_id=self._pad_token_id(),
            eos_token_id=self._eos_token_id(),
        )
        generated_attention = torch.cat(
            [prompt_inputs["attention_mask"], completion_attention],
            dim=1,
        )
        return {
            "generated_ids": generated_ids,
            "generated_attention_mask": generated_attention,
            "completion_ids": completion_ids,
            "completion_attention_mask": completion_attention,
            "completion_loss_mask": completion_attention.to(dtype=torch.bool),
        }

    def _init_vllm(self) -> None:
        if self.vllm_mode == "server":
            try:
                from trl.extras.vllm_client import VLLMClient
                from vllm import SamplingParams
            except ImportError as exc:
                raise ImportError(
                    "vLLM server mode requires trl.extras.vllm_client and vllm. "
                    "Install the repository uv environment or set USE_VLLM=false."
                ) from exc

            self._vllm_sampling_params_cls = SamplingParams
            if self.accelerator.is_main_process:
                if self.vllm_server_base_url:
                    base_url = self.vllm_server_base_url
                else:
                    base_url = f"http://{self.vllm_server_host}:{self.vllm_server_port}"
                self.vllm_client = VLLMClient(
                    base_url=base_url,
                    group_port=self.vllm_server_group_port,
                    connection_timeout=self.vllm_server_timeout,
                )
                self.vllm_client.init_communicator(
                    device=torch.cuda.current_device()
                    if torch.cuda.is_available()
                    else "cpu"
                )
            self.accelerator.wait_for_everyone()
            return
        if self.vllm_mode != "colocate":
            raise ValueError(f"Unknown vLLM mode: {self.vllm_mode!r}")

        try:
            from vllm import LLM, SamplingParams
        except ImportError as exc:
            raise ImportError(
                "vLLM is not installed. Run `uv sync --python 3.11` after adding "
                "vllm to pyproject.toml, or set USE_VLLM=false."
            ) from exc

        if self.accelerator.num_processes % self.vllm_tensor_parallel_size != 0:
            raise ValueError(
                "vllm_tensor_parallel_size must divide the number of training processes."
            )

        self._vllm_sampling_params_cls = SamplingParams

        if self.vllm_tensor_parallel_size > 1:
            # Tensor-parallel vLLM ranks must see the same prompt list, so later
            # rollout code gathers local prompts inside these fixed TP subgroups.
            self.vllm_tp_group, _ = torch.distributed.new_subgroups_by_enumeration(
                [
                    list(
                        range(
                            i * self.vllm_tensor_parallel_size,
                            (i + 1) * self.vllm_tensor_parallel_size,
                        )
                    )
                    for i in range(
                        self.accelerator.num_processes // self.vllm_tensor_parallel_size
                    )
                ]
            )

        # vLLM's external launcher reads the same distributed environment variables
        # as torchrun/deepspeed, so mirror Accelerate's process metadata here.
        os.environ["RANK"] = str(self.accelerator.process_index)
        os.environ["LOCAL_RANK"] = str(self.accelerator.local_process_index)
        os.environ["WORLD_SIZE"] = str(self.accelerator.num_processes)
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", "29500")

        max_num_seqs = self.vllm_max_num_seqs
        if max_num_seqs is None:
            max_num_seqs = (
                self.args.per_device_train_batch_size * self.vllm_tensor_parallel_size
            )

        # max_num_seqs is tied to the local micro-batch by default to avoid vLLM
        # reserving memory for more concurrent rollouts than training can consume.
        self.vllm_engine = LLM(
            model=self.model_name_or_path,
            trust_remote_code=True,
            tokenizer_mode="slow",
            tensor_parallel_size=self.vllm_tensor_parallel_size,
            gpu_memory_utilization=self.vllm_gpu_memory_utilization,
            max_num_seqs=max(1, max_num_seqs),
            max_model_len=self.vllm_max_model_len,
            distributed_executor_backend="external_launcher",
            seed=self.accelerator.process_index // self.vllm_tensor_parallel_size,
            limit_mm_per_prompt={"image": 1},
            mm_processor_kwargs={"use_fast": False},
            disable_custom_all_reduce=self.vllm_disable_custom_all_reduce,
        )
        self.accelerator.wait_for_everyone()

    def _generate_on_policy_vllm(
        self,
        raw_inputs: dict[str, Any],
        prompt_inputs: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        prompt_texts = raw_inputs.get("student_prompt_texts")
        images = raw_inputs.get("student_images")
        if prompt_texts is None or images is None:
            raise KeyError(
                "vLLM rollout requires student_prompt_texts and student_images."
            )

        # vLLM consumes raw prompt text and PIL images; the padded token tensors are
        # reconstructed after generation so the loss path matches HF generation.
        vllm_inputs = [
            {"prompt": prompt, "multi_modal_data": {"image": image}}
            for prompt, image in zip(prompt_texts, images, strict=True)
        ]

        if self.vllm_mode == "server":
            return self._generate_on_policy_vllm_server(vllm_inputs, prompt_inputs)
        if self.vllm_mode != "colocate":
            raise ValueError(f"Unknown vLLM mode: {self.vllm_mode!r}")

        sampling_params = self._vllm_sampling_params(self.max_completion_length)

        if hasattr(self, "vllm_tp_group") and self.vllm_tensor_parallel_size > 1:
            # Each TP shard generates over the gathered batch. Slice back to the
            # local process's completions after vLLM returns token ids.
            original_size = len(vllm_inputs)
            gathered = [None for _ in range(self.vllm_tensor_parallel_size)]
            torch.distributed.all_gather_object(
                gathered,
                vllm_inputs,
                group=self.vllm_tp_group,
            )
            all_vllm_inputs = [item for group_items in gathered for item in group_items]
        else:
            all_vllm_inputs = vllm_inputs

        outputs = self.vllm_engine.generate(
            all_vllm_inputs,
            sampling_params=sampling_params,
            use_tqdm=False,
        )
        completion_ids = [output.outputs[0].token_ids for output in outputs]

        if hasattr(self, "vllm_tp_group") and self.vllm_tensor_parallel_size > 1:
            local_rank = torch.distributed.get_rank(group=self.vllm_tp_group)
            keep = slice(
                local_rank * original_size,
                (local_rank + 1) * original_size,
            )
            completion_ids = completion_ids[keep]

        if not completion_ids or max(len(ids) for ids in completion_ids) == 0:
            raise RuntimeError("vLLM generated zero completion tokens for the batch.")

        return self._vllm_completion_ids_to_tensors(completion_ids, prompt_inputs)

    def _generate_on_policy_vllm_server(
        self,
        vllm_inputs: list[dict[str, Any]],
        prompt_inputs: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            gathered: list[list[dict[str, Any]] | None] = [
                None for _ in range(self.accelerator.num_processes)
            ]
            torch.distributed.all_gather_object(gathered, vllm_inputs)
            gathered_inputs = [items or [] for items in gathered]
        else:
            gathered_inputs = [vllm_inputs]

        local_completion_ids: list[list[int]] | None = None
        if self.accelerator.is_main_process:
            flat_inputs = [item for items in gathered_inputs for item in items]
            prompts = [item["prompt"] for item in flat_inputs]
            images = [
                item.get("multi_modal_data", {}).get("image") for item in flat_inputs
            ]
            completion_ids = self._vllm_server_generate_ids(prompts, images)

            by_rank: list[list[list[int]]] = []
            cursor = 0
            for items in gathered_inputs:
                next_cursor = cursor + len(items)
                by_rank.append(completion_ids[cursor:next_cursor])
                cursor = next_cursor
            payload = by_rank
        else:
            payload = None

        if torch.distributed.is_available() and torch.distributed.is_initialized():
            obj_list = [payload]
            torch.distributed.broadcast_object_list(obj_list, src=0)
            payload = obj_list[0]

        if payload is None:
            raise RuntimeError("vLLM server generation did not return a payload.")
        local_completion_ids = payload[self.accelerator.process_index]
        if len(local_completion_ids) != len(vllm_inputs):
            raise RuntimeError(
                "vLLM server returned an unexpected number of completions for "
                f"rank {self.accelerator.process_index}: "
                f"expected {len(vllm_inputs)}, got {len(local_completion_ids)}."
            )
        if not local_completion_ids or max(len(ids) for ids in local_completion_ids) == 0:
            raise RuntimeError("vLLM server generated zero completion tokens for the batch.")

        return self._vllm_completion_ids_to_tensors(local_completion_ids, prompt_inputs)

    def _vllm_server_generate_ids(
        self,
        prompts: list[str],
        images: list[Any],
    ) -> list[list[int]]:
        if not prompts:
            return []
        top_k = self.generation_top_k if self.generation_top_k > 0 else -1
        generation_kwargs: dict[str, Any] = {}
        if self.presence_penalty != 0.0:
            generation_kwargs["presence_penalty"] = self.presence_penalty

        request_batch_size = self.vllm_server_request_batch_size or len(prompts)
        completion_ids: list[list[int]] = []
        for start in range(0, len(prompts), request_batch_size):
            end = start + request_batch_size
            output = self.vllm_client.generate(
                prompts=prompts[start:end],
                images=images[start:end],
                n=1,
                repetition_penalty=self.repetition_penalty,
                temperature=self.generation_temperature,
                top_p=self.generation_top_p,
                top_k=top_k,
                min_p=self.min_p,
                max_tokens=self.max_completion_length,
                generation_kwargs=generation_kwargs,
            )
            completion_ids.extend(output["completion_ids"])
        return completion_ids

    def _vllm_completion_ids_to_tensors(
        self,
        completion_ids: list[list[int]],
        prompt_inputs: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        if not completion_ids or max(len(ids) for ids in completion_ids) == 0:
            raise RuntimeError("vLLM generated zero completion tokens for the batch.")

        pad_token_id = self._pad_token_id()
        if pad_token_id is None:
            pad_token_id = self._eos_token_id()
        if pad_token_id is None:
            raise RuntimeError(
                "A pad or EOS token is required to pad vLLM completions."
            )

        # Convert ragged vLLM outputs into dense tensors plus an attention mask, the
        # same contract returned by the HF generation path above.
        max_len = min(
            self.max_completion_length,
            max(1, max(len(ids) for ids in completion_ids)),
        )
        completion_lengths = [min(len(ids), max_len) for ids in completion_ids]
        completion_tensor = torch.full(
            (len(completion_ids), max_len),
            pad_token_id,
            dtype=prompt_inputs["input_ids"].dtype,
            device=prompt_inputs["input_ids"].device,
        )
        for row, ids in enumerate(completion_ids):
            clipped = ids[:max_len]
            if clipped:
                completion_tensor[row, : len(clipped)] = torch.tensor(
                    clipped,
                    dtype=completion_tensor.dtype,
                    device=completion_tensor.device,
                )

        completion_attention = self._completion_attention_from_lengths(
            completion_lengths,
            max_len,
            device=completion_tensor.device,
        )
        generated_ids = torch.cat(
            [prompt_inputs["input_ids"], completion_tensor], dim=1
        )
        generated_attention = torch.cat(
            [prompt_inputs["attention_mask"], completion_attention],
            dim=1,
        )
        return {
            "generated_ids": generated_ids,
            "generated_attention_mask": generated_attention,
            "completion_ids": completion_tensor,
            "completion_attention_mask": completion_attention,
            "completion_loss_mask": completion_attention.to(dtype=torch.bool),
        }

    def _vllm_sampling_params(self, max_tokens: int):
        top_k = self.generation_top_k if self.generation_top_k > 0 else -1
        return self._vllm_sampling_params_cls(
            n=1,
            repetition_penalty=self.repetition_penalty,
            temperature=self.generation_temperature,
            top_p=self.generation_top_p,
            top_k=top_k,
            min_p=self.min_p,
            max_tokens=max(1, max_tokens),
            presence_penalty=self.presence_penalty,
        )

    def _move_model_to_vllm(self) -> None:
        if not self.use_vllm:
            return
        unwrapped = self.accelerator.unwrap_model(self.model)

        if is_peft_model(unwrapped):
            # vLLM samples from merged base+LoRA weights. Merge only inside the
            # loading window, then unmerge so training continues on adapter params.
            with self._zero3_gather_context(list(unwrapped.parameters())):
                unwrapped.merge_adapter()
                try:
                    self._load_weights_into_vllm(unwrapped)
                finally:
                    unwrapped.unmerge_adapter()
        elif self.is_fsdp_enabled:
            self._load_weights_into_vllm(unwrapped)
        else:
            for name, param in unwrapped.named_parameters():
                with self._zero3_gather_context([param]):
                    self._load_named_weight_into_vllm(name, param.data)

        if self.vllm_mode == "server":
            if self.accelerator.is_main_process:
                self.vllm_client.reset_prefix_cache()
            self.accelerator.wait_for_everyone()
            return
        if self.vllm_mode != "colocate":
            raise ValueError(f"Unknown vLLM mode: {self.vllm_mode!r}")
        self.vllm_engine.reset_prefix_cache()
        self.accelerator.wait_for_everyone()

    def _load_weights_into_vllm(self, model: nn.Module) -> None:
        for name, param in model.named_parameters():
            mapped_name = self._map_weight_name_for_vllm(name, model)
            if mapped_name is not None:
                self._load_named_weight_into_vllm(mapped_name, param.data)

    def _load_named_weight_into_vllm(self, name: str, weight: torch.Tensor) -> None:
        if self.vllm_mode == "server":
            if self.accelerator.is_main_process:
                self.vllm_client.update_named_param(
                    self._map_weight_name_for_vllm(name, self.accelerator.unwrap_model(self.model))
                    or name,
                    weight.contiguous(),
                )
            return
        if self.vllm_mode != "colocate":
            raise ValueError(f"Unknown vLLM mode: {self.vllm_mode!r}")
        llm_model = (
            self.vllm_engine.llm_engine.model_executor.driver_worker.model_runner.model
        )
        llm_model.load_weights([(name, weight)])

    @staticmethod
    def _map_weight_name_for_vllm(name: str, model: nn.Module) -> str | None:
        mapped_name = name.removeprefix("base_model.model.").replace(".base_layer", "")
        prefix = getattr(model, "prefix", None)
        if is_peft_model(model) and prefix and prefix in mapped_name:
            return None
        if "original_module" in mapped_name:
            return None
        return mapped_name.replace("modules_to_save.default.", "")

    def _zero3_gather_context(self, params: list[torch.Tensor]):
        deepspeed_plugin = getattr(self.accelerator.state, "deepspeed_plugin", None)
        if (
            deepspeed_plugin is not None
            and getattr(deepspeed_plugin, "zero_stage", None) == 3
        ):
            import deepspeed

            return deepspeed.zero.GatheredParameters(params)
        return nullcontext()

    def _maybe_add_generation_arg(
        self,
        model: nn.Module,
        generation_kwargs: dict[str, Any],
        name: str,
        value: float,
        active: bool,
    ) -> None:
        if not active:
            return
        config = getattr(model, "generation_config", None)
        try:
            accepts_kwarg = name in inspect.signature(model.generate).parameters
        except (TypeError, ValueError):
            accepts_kwarg = False
        if accepts_kwarg or (config is not None and hasattr(config, name)):
            generation_kwargs[name] = value
            return
        if name not in self._warned_unsupported_generation_args:
            self._warned_unsupported_generation_args.add(name)
            if self.accelerator.is_main_process:
                print(
                    f"[ViGOSTrainer] Skipping unsupported HF generation argument "
                    f"{name}={value!r}. Use vLLM for this sampling control."
                )

    def _teacher_context(self, model: nn.Module):
        if not self.fixed_teacher:
            return nullcontext()
        return self._disable_adapter_context(model)

    def _disable_adapter_context(self, model: nn.Module):
        unwrapped = self._unwrap_train_model(model)
        # Disabling PEFT adapters turns the current LoRA student into a frozen-base
        # teacher for the perception and privileged reasoning scoring passes.
        if is_peft_model(unwrapped):
            return unwrapped.disable_adapter()
        return nullcontext()

    @contextmanager
    def _temporary_eval_context(self, model: nn.Module):
        unwrapped = self._unwrap_train_model(model)
        module_training_states = [
            (module, module.training) for module in unwrapped.modules()
        ]
        try:
            unwrapped.eval()
            yield
        finally:
            for module, training in module_training_states:
                module.training = training

    def _unwrap_train_model(self, model: nn.Module) -> nn.Module:
        accelerator = getattr(self, "accelerator", None)
        if accelerator is None:
            return model
        return accelerator.unwrap_model(model)

    @staticmethod
    def _prompt_inputs(inputs: dict[str, Any], prefix: str) -> dict[str, torch.Tensor]:
        marker = f"{prefix}_prompt_"
        selected = {
            key.removeprefix(marker): value
            for key, value in inputs.items()
            if key.startswith(marker) and key.removeprefix(marker) in MODEL_INPUT_KEYS
        }
        if "input_ids" not in selected or "attention_mask" not in selected:
            raise KeyError(f"Missing input_ids or attention_mask for {prefix} prompt.")
        return selected

    @staticmethod
    def _with_completion(
        prompt_inputs: dict[str, torch.Tensor],
        full_input_ids: torch.Tensor,
        full_attention_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        result = {
            key: value
            for key, value in prompt_inputs.items()
            if key not in {"input_ids", "attention_mask"}
        }
        result["input_ids"] = full_input_ids
        result["attention_mask"] = full_attention_mask
        return result

    @staticmethod
    def _append_completion(
        prompt_inputs: dict[str, torch.Tensor],
        completion_ids: torch.Tensor,
        completion_attention: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        full_input_ids = torch.cat([prompt_inputs["input_ids"], completion_ids], dim=1)
        full_attention_mask = torch.cat(
            [prompt_inputs["attention_mask"], completion_attention], dim=1
        )
        return ViGOSTrainer._with_completion(
            prompt_inputs, full_input_ids, full_attention_mask
        )

    def _batched_teacher_completion_logits(
        self,
        model: nn.Module,
        jobs: list[dict[str, Any]],
    ) -> dict[str, torch.Tensor]:
        if not jobs:
            return {}
        merged_inputs, batch_slices = self._merge_model_inputs(
            [job["inputs"] for job in jobs]
        )
        max_logits_to_keep = max(
            int(job["completion_length"]) + 1 for job in jobs
        )
        merged_inputs["logits_to_keep"] = max_logits_to_keep
        with (
            torch.no_grad(),
            self._temporary_eval_context(model),
            self._teacher_context(model),
        ):
            outputs = model(**merged_inputs)
            logits = outputs.logits.detach()
            del outputs

        result = {}
        for job, (start, end) in zip(jobs, batch_slices, strict=True):
            completion_length = int(job["completion_length"])
            result[str(job["name"])] = self._completion_logits(
                logits[start:end],
                completion_length,
            ).detach()
        return result

    def _merge_model_inputs(
        self,
        inputs_list: list[dict[str, torch.Tensor]],
    ) -> tuple[dict[str, torch.Tensor], list[tuple[int, int]]]:
        if not inputs_list:
            raise ValueError("Cannot merge an empty model-input list.")

        pad_token_id = self._pad_token_id()
        if pad_token_id is None:
            pad_token_id = self._eos_token_id()
        if pad_token_id is None:
            pad_token_id = 0

        max_length = max(int(inputs["input_ids"].shape[1]) for inputs in inputs_list)
        input_rows = []
        attention_rows = []
        batch_slices: list[tuple[int, int]] = []
        cursor = 0
        for inputs in inputs_list:
            input_ids = inputs["input_ids"]
            attention_mask = inputs["attention_mask"]
            pad_width = max_length - int(input_ids.shape[1])
            if pad_width > 0:
                input_pad = input_ids.new_full(
                    (input_ids.shape[0], pad_width),
                    int(pad_token_id),
                )
                attention_pad = attention_mask.new_zeros(
                    (attention_mask.shape[0], pad_width)
                )
                input_ids = torch.cat([input_pad, input_ids], dim=1)
                attention_mask = torch.cat([attention_pad, attention_mask], dim=1)
            input_rows.append(input_ids)
            attention_rows.append(attention_mask)
            next_cursor = cursor + int(input_ids.shape[0])
            batch_slices.append((cursor, next_cursor))
            cursor = next_cursor

        merged: dict[str, torch.Tensor] = {
            "input_ids": torch.cat(input_rows, dim=0),
            "attention_mask": torch.cat(attention_rows, dim=0),
        }
        optional_keys = sorted(
            {
                key
                for inputs in inputs_list
                for key in inputs
                if key not in {"input_ids", "attention_mask", "logits_to_keep"}
            }
        )
        for key in optional_keys:
            values = []
            for inputs in inputs_list:
                if key not in inputs:
                    raise ValueError(
                        f"Cannot merge model inputs with missing key {key!r}."
                    )
                value = inputs[key]
                if not isinstance(value, torch.Tensor):
                    raise TypeError(
                        f"Cannot merge non-tensor model input {key!r}: "
                        f"{type(value)!r}."
                    )
                values.append(value)
            merged[key] = torch.cat(values, dim=0)
        return merged, batch_slices

    @staticmethod
    def _completion_attention_from_token_ids(
        completion_ids: torch.Tensor,
        pad_token_id: int | None,
        eos_token_id: int | list[int] | tuple[int, ...] | None,
    ) -> torch.Tensor:
        # HF generate may append PAD after EOS. Keep tokens through the first EOS and
        # mask the padded tail so Perception/Reasoning do not train on artificial padding.
        attention = torch.ones_like(completion_ids, dtype=torch.long)
        eos_token_ids = ViGOSTrainer._normalize_eos_token_ids(eos_token_id)
        for row_idx in range(completion_ids.shape[0]):
            row = completion_ids[row_idx]
            valid_length = row.shape[0]
            eos_positions = ViGOSTrainer._token_positions(row, eos_token_ids)
            if eos_positions.numel() > 0:
                valid_length = int(eos_positions[0].item()) + 1
            elif pad_token_id is not None:
                pad_positions = (row == pad_token_id).nonzero(as_tuple=False).flatten()
                if pad_positions.numel() > 0:
                    valid_length = int(pad_positions[0].item())
            if valid_length < row.shape[0]:
                attention[row_idx, valid_length:] = 0
        return attention

    @staticmethod
    def _completion_attention_from_lengths(
        lengths: list[int],
        max_length: int,
        device: torch.device,
    ) -> torch.Tensor:
        positions = torch.arange(max_length, device=device).unsqueeze(0)
        length_tensor = torch.tensor(
            lengths, dtype=torch.long, device=device
        ).unsqueeze(1)
        return positions.lt(length_tensor).to(dtype=torch.long)

    @staticmethod
    def _completion_logits(
        logits: torch.Tensor, completion_length: int
    ) -> torch.Tensor:
        # For causal LM scoring, logits at position t predict token t+1; the final
        # prompt logit plus completion logits score the generated completion tokens.
        needed_positions = completion_length + 1
        if logits.shape[1] < needed_positions:
            raise RuntimeError(
                "Model returned too few logit positions for completion scoring: "
                f"got {logits.shape[1]}, need at least {needed_positions}."
            )
        return logits[:, -needed_positions:-1, :]

    @staticmethod
    def _normalize_eos_token_ids(
        eos_token_id: int | list[int] | tuple[int, ...] | None,
    ) -> tuple[int, ...]:
        if eos_token_id is None:
            return ()
        if isinstance(eos_token_id, int):
            return (eos_token_id,)
        return tuple(int(token_id) for token_id in eos_token_id)

    @staticmethod
    def _token_positions(row: torch.Tensor, token_ids: tuple[int, ...]) -> torch.Tensor:
        if not token_ids:
            return row.new_empty((0,), dtype=torch.long)
        matches = torch.zeros_like(row, dtype=torch.bool)
        for token_id in token_ids:
            matches |= row == token_id
        return matches.nonzero(as_tuple=False).flatten()

    def _pad_token_id(self) -> int | None:
        tokenizer = getattr(self.processor, "tokenizer", self.processor)
        return getattr(tokenizer, "pad_token_id", None)

    def _eos_token_id(self) -> int | None:
        tokenizer = getattr(self.processor, "tokenizer", self.processor)
        return getattr(tokenizer, "eos_token_id", None)

def _casefold_answer(value: str) -> str:
    return " ".join(str(value).strip().casefold().split())
