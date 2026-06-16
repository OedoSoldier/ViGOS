"""ViGOS training entry point."""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

if __package__ is None or __package__ == "":
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoConfig,
    AutoProcessor,
    HfArgumentParser,
    TrainerCallback,
    TrainingArguments,
    set_seed,
)

import vigos.dataset_utils as dataset_utils
from vigos.data_collator import ViGOSDataCollator
from vigos.trainer import ViGOSTrainer

_load_dataset = dataset_utils.load_vigos_dataset
_filter_tiny_image_samples = dataset_utils.filter_tiny_image_samples

DEFAULT_LEARNING_RATE = 5e-6

try:
    from transformers import Qwen2_5_VLForConditionalGeneration
    from transformers import Qwen3VLForConditionalGeneration
except ImportError as exc:  # pragma: no cover - depends on installed transformers version.
    raise ImportError(
        "Qwen2.5-VL or Qwen3-VL model classes are unavailable. Install the uv "
        "environment with transformers==4.57.1 or newer VL model support."
    ) from exc


@dataclass
class ViGOSScriptArguments:
    model_name_or_path: str = "Qwen/Qwen2.5-VL-3B-Instruct"
    dataset_name: str | None = None
    dataset_split: str = "train"
    max_train_samples: Optional[int] = None
    filter_tiny_images: bool = False
    min_image_size: int = 3
    max_prompt_length: int = 32768
    max_completion_length: int = 4096
    attn_implementation: str = "flash_attention_2"
    torch_dtype: str = "bfloat16"
    trust_remote_code: bool = True
    lora_r: int = 64
    lora_alpha: int = 128
    lora_dropout: float = 0.05
    lora_target_modules: str = "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj"
    answer_field: str = "answer"
    generation_temperature: float = 1.1
    generation_top_p: float = 0.95
    generation_top_k: int = 20
    distill_temperature: float = 1.0
    lambda_perception: float = 1.0
    lambda_reasoning: float = 1.0
    lambda_ref: float = 2.0
    token_loss_clip: float = 0.05
    description_last_token_clip: float = 0.05
    reasoning_first_token_clip: float = 0.05
    presence_penalty: float = 0.0
    repetition_penalty: float = 1.0
    min_p: float = 0.0
    use_vllm: bool = False
    vllm_mode: str = "colocate"
    vllm_gpu_memory_utilization: float = 0.6
    vllm_tensor_parallel_size: int = 1
    vllm_sync_frequency: int = 1
    vllm_max_model_len: Optional[int] = None
    vllm_max_num_seqs: Optional[int] = None
    vllm_disable_custom_all_reduce: bool = False
    vllm_server_base_url: Optional[str] = None
    vllm_server_host: str = "127.0.0.1"
    vllm_server_port: int = 8000
    vllm_server_timeout: float = 300.0
    vllm_server_group_port: int = 51216
    vllm_server_request_batch_size: Optional[int] = None
    completion_log_steps: int = 2
    completion_log_max_samples: int = 16
    run_config: str | None = field(
        default=None,
        metadata={
            "help": "Run config label. Appended to output_dir and used for WandB run name."
        },
    )
    run_name_suffix: str | None = field(
        default=None,
        metadata={"help": "Optional suffix appended to the Trainer run name."},
    )


def main() -> None:
    parser = HfArgumentParser((ViGOSScriptArguments, TrainingArguments))
    script_args, training_args = parser.parse_args_into_dataclasses()
    training_args.remove_unused_columns = False
    if not _cli_arg_was_provided("--learning_rate", "--learning-rate"):
        training_args.learning_rate = DEFAULT_LEARNING_RATE

    if not script_args.dataset_name:
        raise ValueError(
            "--dataset_name is required and must point to a HuggingFace dataset id."
        )

    # Mirror experiment names in both Trainer/WandB and the output path so logs,
    # checkpoints, and records can be traced back to the exact run config.
    if script_args.run_config:
        lr_str = f"{training_args.learning_rate:.0e}".replace("e-0", "e-")
        world_size = int(os.environ.get("WORLD_SIZE", "1"))
        effective_batch_size = (
            training_args.per_device_train_batch_size
            * training_args.gradient_accumulation_steps
            * world_size
        )
        training_args.run_name = (
            f"{script_args.run_config}_lr{lr_str}_bs{effective_batch_size}"
        )
        if not Path(training_args.output_dir).name == script_args.run_config:
            training_args.output_dir = str(Path(training_args.output_dir) / script_args.run_config)
    elif script_args.run_name_suffix:
        base_name = training_args.run_name or "vigos"
        training_args.run_name = f"{base_name}_{script_args.run_name_suffix}"
    elif not training_args.run_name or training_args.run_name == training_args.output_dir:
        training_args.run_name = os.path.basename(os.path.normpath(training_args.output_dir)) or "vigos"

    if os.environ.get("LOCAL_RANK", "0") == "0":
        print("\n" + "=" * 80)
        print("ViGOS RUN CONFIGURATION")
        print("=" * 80)
        print(f"Model path: {script_args.model_name_or_path}")
        print(f"WandB/Trainer run name: {training_args.run_name}")
        print(f"Output directory: {training_args.output_dir}")
        print(f"Dataset name: {script_args.dataset_name}")
        print(
            "Dataset filtering: "
            f"filter_tiny_images={script_args.filter_tiny_images}, "
            f"min_image_size={script_args.min_image_size}"
        )
        print(f"Answer/reference field: {script_args.answer_field}")
        print(
            "Rollout sampling: "
            f"temperature={script_args.generation_temperature}, "
            f"top_p={script_args.generation_top_p}, "
            f"top_k={script_args.generation_top_k}, "
            f"presence_penalty={script_args.presence_penalty}, "
            f"repetition_penalty={script_args.repetition_penalty}, "
            f"min_p={script_args.min_p}"
        )
        print(
            "Distillation: "
            f"lambda_perception={script_args.lambda_perception}, "
            f"lambda_reasoning={script_args.lambda_reasoning}, "
            f"lambda_ref={script_args.lambda_ref}, "
            "fallback=reference_only, "
            "ref_kl_direction=reverse, "
            f"distill_temperature={script_args.distill_temperature}, "
            f"token_loss_clip={script_args.token_loss_clip}, "
            f"description_last_token_clip={script_args.description_last_token_clip}, "
            f"reasoning_first_token_clip={script_args.reasoning_first_token_clip}"
        )
        print("Malformed tags: apply Reference loss over the full completion.")
        print(
            "Completion snapshots: "
            f"steps={script_args.completion_log_steps}, "
            f"max_samples={script_args.completion_log_max_samples}"
        )
        print(
            "vLLM: "
            f"use_vllm={script_args.use_vllm}, "
            f"mode={script_args.vllm_mode}, "
            f"tp={script_args.vllm_tensor_parallel_size}, "
            f"gpu_memory_utilization={script_args.vllm_gpu_memory_utilization}, "
            f"max_num_seqs={script_args.vllm_max_num_seqs}, "
            f"disable_custom_all_reduce={script_args.vllm_disable_custom_all_reduce}, "
            f"server_base_url={script_args.vllm_server_base_url}, "
            f"server_host={script_args.vllm_server_host}, "
            f"server_port={script_args.vllm_server_port}, "
            f"server_request_batch_size={script_args.vllm_server_request_batch_size}"
        )
        print("=" * 80 + "\n")

    set_seed(training_args.seed)

    # Qwen VL processors wrap the tokenizer. Left padding keeps the prompt suffix
    # aligned for decoder-only generation when batches contain different lengths.
    processor = AutoProcessor.from_pretrained(
        script_args.model_name_or_path,
        trust_remote_code=script_args.trust_remote_code,
        use_fast=False,
    )
    tokenizer = getattr(processor, "tokenizer", processor)
    if getattr(tokenizer, "pad_token", None) is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model_kwargs = {
        "trust_remote_code": script_args.trust_remote_code,
        "attn_implementation": script_args.attn_implementation,
        "dtype": _dtype(script_args.torch_dtype),
    }
    # Dispatch from config.model_type so Qwen2.5-VL and Qwen3-VL checkpoints use
    # their native multimodal classes instead of a text-only AutoModel fallback.
    model_class, model_type = _model_class_for_checkpoint(
        script_args.model_name_or_path,
        trust_remote_code=script_args.trust_remote_code,
    )
    if os.environ.get("LOCAL_RANK", "0") == "0":
        print(f"Resolved model_type={model_type} to {model_class.__name__}.")
    model = model_class.from_pretrained(script_args.model_name_or_path, **model_kwargs)
    model.config.use_cache = False if training_args.gradient_checkpointing else True

    # Training with checkpointed activations cannot reuse generation KV cache during
    # the forward/backward pass, so keep use_cache disabled for the train model.
    if training_args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.config.use_cache = False

    # ViGOS trains LoRA adapters only; this keeps checkpoints small and lets merge
    # scripts combine the adapter with the selected base model after training.
    target_modules = [
        module.strip()
        for module in script_args.lora_target_modules.split(",")
        if module.strip()
    ]
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=script_args.lora_r,
        lora_alpha=script_args.lora_alpha,
        lora_dropout=script_args.lora_dropout,
        target_modules=target_modules,
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # The collator builds all ViGOS prompt tensors: student, image-only
    # perception, privileged reasoning, and fixed Reference teacher prompts.
    # The trainer appends on-policy rollout spans to those prompt tensors.
    dataset = _load_dataset(script_args.dataset_name, script_args.dataset_split)
    raw_dataset_size = len(dataset)
    if script_args.max_train_samples is not None:
        dataset = dataset.select(range(min(script_args.max_train_samples, len(dataset))))
    pre_filter_dataset_size = len(dataset)
    dataset_filter_removed = 0
    if script_args.filter_tiny_images:
        dataset = _filter_tiny_image_samples(
            dataset,
            min_image_size=script_args.min_image_size,
        )
        dataset_filter_removed = pre_filter_dataset_size - len(dataset)
        if os.environ.get("LOCAL_RANK", "0") == "0":
            print(
                "Dataset filtering removed "
                f"{dataset_filter_removed}/{pre_filter_dataset_size} samples with "
                f"missing, unreadable, or smaller-than-{script_args.min_image_size}px images."
            )

    data_collator = ViGOSDataCollator(
        processor=processor,
        max_prompt_length=script_args.max_prompt_length,
        answer_field=script_args.answer_field,
    )

    # ViGOSTrainer owns the on-policy rollout and Perception/Reasoning loss computation; the
    # entry point only wires model, data, and reproducible hyperparameters together.
    trainer = ViGOSTrainer(
        model=model,
        args=training_args,
        model_name_or_path=script_args.model_name_or_path,
        train_dataset=dataset,
        data_collator=data_collator,
        processing_class=processor,
        processor=processor,
        max_prompt_length=script_args.max_prompt_length,
        max_completion_length=script_args.max_completion_length,
        generation_temperature=script_args.generation_temperature,
        generation_top_p=script_args.generation_top_p,
        generation_top_k=script_args.generation_top_k,
        distill_temperature=script_args.distill_temperature,
        lambda_perception=script_args.lambda_perception,
        lambda_reasoning=script_args.lambda_reasoning,
        lambda_ref=script_args.lambda_ref,
        token_loss_clip=(
            script_args.token_loss_clip if script_args.token_loss_clip > 0 else None
        ),
        description_last_token_clip=(
            script_args.description_last_token_clip
            if script_args.description_last_token_clip > 0
            else None
        ),
        reasoning_first_token_clip=(
            script_args.reasoning_first_token_clip
            if script_args.reasoning_first_token_clip > 0
            else None
        ),
        presence_penalty=script_args.presence_penalty,
        repetition_penalty=script_args.repetition_penalty,
        min_p=script_args.min_p,
        use_vllm=script_args.use_vllm,
        vllm_mode=script_args.vllm_mode,
        vllm_gpu_memory_utilization=script_args.vllm_gpu_memory_utilization,
        vllm_tensor_parallel_size=script_args.vllm_tensor_parallel_size,
        vllm_sync_frequency=script_args.vllm_sync_frequency,
        vllm_max_model_len=script_args.vllm_max_model_len
        or script_args.max_prompt_length + script_args.max_completion_length,
        vllm_max_num_seqs=script_args.vllm_max_num_seqs,
        vllm_disable_custom_all_reduce=script_args.vllm_disable_custom_all_reduce,
        vllm_server_base_url=script_args.vllm_server_base_url,
        vllm_server_host=script_args.vllm_server_host,
        vllm_server_port=script_args.vllm_server_port,
        vllm_server_timeout=script_args.vllm_server_timeout,
        vllm_server_group_port=script_args.vllm_server_group_port,
        vllm_server_request_batch_size=script_args.vllm_server_request_batch_size,
        completion_log_steps=script_args.completion_log_steps,
        completion_log_max_samples=script_args.completion_log_max_samples,
    )
    if _reporting_to_wandb(training_args):
        # Trainer.log records dynamic losses; this callback adds static run metadata
        # to WandB so runs can be filtered by model family and ViGOS settings.
        trainer.add_callback(
            WandBConfigCallback(
                _wandb_config(
                    script_args,
                    training_args,
                    train_dataset_size=len(dataset),
                    raw_train_dataset_size=raw_dataset_size,
                    pre_filter_train_dataset_size=pre_filter_dataset_size,
                    filtered_train_dataset_size=len(dataset),
                    dataset_filter_removed=dataset_filter_removed,
                )
            )
        )
    trainer.train()
    # With PEFT enabled, save_model writes LoRA adapter weights rather than a full
    # base checkpoint. The processor is saved beside the adapter for merge/eval use.
    trainer.save_model(training_args.output_dir)
    processor.save_pretrained(training_args.output_dir)


def _dtype(name: str) -> torch.dtype:
    normalized = name.lower()
    if normalized in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if normalized in {"fp16", "float16"}:
        return torch.float16
    if normalized in {"fp32", "float32"}:
        return torch.float32
    raise ValueError(f"Unsupported torch dtype: {name}")


def _model_class_for_checkpoint(
    model_name_or_path: str,
    trust_remote_code: bool = True,
) -> tuple[type[torch.nn.Module], str]:
    config = AutoConfig.from_pretrained(
        model_name_or_path,
        trust_remote_code=trust_remote_code,
    )
    model_type = getattr(config, "model_type", "")
    if model_type == "qwen2_5_vl":
        return Qwen2_5_VLForConditionalGeneration, model_type
    if model_type == "qwen3_vl":
        return Qwen3VLForConditionalGeneration, model_type
    raise ValueError(
        "Unsupported ViGOS model_type. Expected one of {'qwen2_5_vl', 'qwen3_vl'}, "
        f"got {model_type!r} from {model_name_or_path!r}."
    )


def _reporting_to_wandb(training_args: TrainingArguments) -> bool:
    report_to = training_args.report_to
    if report_to is None:
        return False
    if isinstance(report_to, str):
        targets = [report_to]
    else:
        targets = list(report_to)
    return any(str(target).lower() == "wandb" for target in targets)


def _cli_arg_was_provided(*option_names: str) -> bool:
    for arg in sys.argv[1:]:
        for name in option_names:
            if arg == name or arg.startswith(f"{name}="):
                return True
    return False


def _wandb_config(
    script_args: ViGOSScriptArguments,
    training_args: TrainingArguments,
    train_dataset_size: int,
    raw_train_dataset_size: int,
    pre_filter_train_dataset_size: int,
    filtered_train_dataset_size: int,
    dataset_filter_removed: int,
) -> dict[str, object]:
    # Store derived defaults as resolved values, not only CLI inputs, to make WandB
    # records sufficient for reproducing a run from scratch.
    vllm_max_model_len = (
        script_args.vllm_max_model_len
        or script_args.max_prompt_length + script_args.max_completion_length
    )
    vllm_max_num_seqs = (
        script_args.vllm_max_num_seqs
        or training_args.per_device_train_batch_size * script_args.vllm_tensor_parallel_size
    )
    return {
        "vigos_model_name_or_path": script_args.model_name_or_path,
        "vigos_model_family": _model_type_for_config_path(
            script_args.model_name_or_path,
            trust_remote_code=script_args.trust_remote_code,
        ),
        "vigos_dataset_name": script_args.dataset_name,
        "vigos_dataset_split": script_args.dataset_split,
        "vigos_train_dataset_size": train_dataset_size,
        "vigos_raw_train_dataset_size": raw_train_dataset_size,
        "vigose_filter_train_dataset_size": pre_filter_train_dataset_size,
        "vigos_filtered_train_dataset_size": filtered_train_dataset_size,
        "vigos_dataset_filter_removed": dataset_filter_removed,
        "vigos_filter_tiny_images": script_args.filter_tiny_images,
        "vigos_min_image_size": script_args.min_image_size,
        "vigos_max_train_samples": script_args.max_train_samples,
        "vigos_max_prompt_length": script_args.max_prompt_length,
        "vigos_max_completion_length": script_args.max_completion_length,
        "vigos_method": "vigos",
        "vigos_answer_field": script_args.answer_field,
        "vigos_fixed_teacher": True,
        "vigos_generation_temperature": script_args.generation_temperature,
        "vigos_generation_top_p": script_args.generation_top_p,
        "vigos_generation_top_k": script_args.generation_top_k,
        "vigos_lambda_perception": script_args.lambda_perception,
        "vigos_lambda_reasoning": script_args.lambda_reasoning,
        "vigos_lambda_ref": script_args.lambda_ref,
        "vigos_malformed_output_loss": "reference_reverse_kl_full_completion",
        "vigos_reference_loss": "reverse_kl",
        "vigos_distill_temperature": script_args.distill_temperature,
        "vigos_token_loss_clip": (
            script_args.token_loss_clip if script_args.token_loss_clip > 0 else None
        ),
        "vigos_description_last_token_clip": (
            script_args.description_last_token_clip
            if script_args.description_last_token_clip > 0
            else None
        ),
        "vigos_reasoning_first_token_clip": (
            script_args.reasoning_first_token_clip
            if script_args.reasoning_first_token_clip > 0
            else None
        ),
        "vigosesence_penalty": script_args.presence_penalty,
        "vigos_repetition_penalty": script_args.repetition_penalty,
        "vigos_min_p": script_args.min_p,
        "vigos_run_config": script_args.run_config,
        "vigos_completion_log_steps": script_args.completion_log_steps,
        "vigos_completion_log_max_samples": script_args.completion_log_max_samples,
        "vigos_use_vllm": script_args.use_vllm,
        "vigos_vllm_mode": script_args.vllm_mode if script_args.use_vllm else None,
        "vigos_vllm_gpu_memory_utilization": (
            script_args.vllm_gpu_memory_utilization if script_args.use_vllm else None
        ),
        "vigos_vllm_tensor_parallel_size": (
            script_args.vllm_tensor_parallel_size if script_args.use_vllm else None
        ),
        "vigos_vllm_sync_frequency": (
            script_args.vllm_sync_frequency if script_args.use_vllm else None
        ),
        "vigos_vllm_max_model_len": vllm_max_model_len if script_args.use_vllm else None,
        "vigos_vllm_max_num_seqs": (
            vllm_max_num_seqs if script_args.use_vllm else None
        ),
        "vigos_vllm_disable_custom_all_reduce": (
            script_args.vllm_disable_custom_all_reduce if script_args.use_vllm else None
        ),
        "vigos_vllm_server_base_url": (
            script_args.vllm_server_base_url if script_args.use_vllm else None
        ),
        "vigos_vllm_server_host": (
            script_args.vllm_server_host if script_args.use_vllm else None
        ),
        "vigos_vllm_server_port": (
            script_args.vllm_server_port if script_args.use_vllm else None
        ),
        "vigos_vllm_server_timeout": (
            script_args.vllm_server_timeout if script_args.use_vllm else None
        ),
        "vigos_vllm_server_group_port": (
            script_args.vllm_server_group_port if script_args.use_vllm else None
        ),
        "vigos_vllm_server_request_batch_size": (
            script_args.vllm_server_request_batch_size if script_args.use_vllm else None
        ),
        "vigos_use_peft": True,
        "vigos_lora_r": script_args.lora_r,
        "vigos_lora_alpha": script_args.lora_alpha,
        "vigos_lora_dropout": script_args.lora_dropout,
        "vigos_lora_target_modules": script_args.lora_target_modules,
        "vigos_attn_implementation": script_args.attn_implementation,
        "vigos_torch_dtype": script_args.torch_dtype,
        "vigos_output_dir": training_args.output_dir,
        "vigos_run_name": training_args.run_name,
    }


def _model_type_for_config_path(
    model_name_or_path: str,
    trust_remote_code: bool = True,
) -> str:
    return getattr(
        AutoConfig.from_pretrained(
            model_name_or_path,
            trust_remote_code=trust_remote_code,
        ),
        "model_type",
        "unknown",
    )


class WandBConfigCallback(TrainerCallback):
    def __init__(self, config: dict[str, object]):
        self.config = config

    def on_train_begin(self, args, state, control, **kwargs):
        if not getattr(state, "is_world_process_zero", True):
            return control
        try:
            import wandb
        except ImportError:
            return control
        if wandb.run is not None:
            wandb.config.update(self.config, allow_val_change=True)
        return control


if __name__ == "__main__":
    main()
