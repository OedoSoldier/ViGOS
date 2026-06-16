#!/usr/bin/env python3
"""Run independent ViGOS evaluation with configurable judging."""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable

from transformers import AutoProcessor

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from vigos.eval_utils import (  # noqa: E402
    DEFAULT_ZLI_DATASETS,
    DEFAULT_EVAL_PROMPT_MODE,
    VIGOS_PROMPT_MODE,
    assistant_prefill_for_prompt_mode,
    build_eval_prompt,
    build_judge_messages,
    build_passk_judge_messages,
    extract_model_answer,
    normalize_eval_prompt_mode,
    parse_dataset_specs,
    parse_judge_output,
    prompt_mode_description,
    response_from_completion,
    sample_from_record,
    sanitize_dataset_name,
    vllm_request,
)
from vigos.eval_benchmarks import (  # noqa: E402
    DEFAULT_EVAL_BENCHMARKS,
    benchmark_response_stem,
    load_benchmark_tasks,
    parse_benchmark_specs,
    score_benchmark,
)


DEFAULT_JUDGE_PROXY = ""
DEFAULT_JUDGE_API_URL = "https://api.deepseek.com"
DEFAULT_JUDGE_MODEL = "deepseek-v4-flash"
DEFAULT_JUDGE_KEY_ENV = "DEEPSEEK_API_KEY"
DEFAULT_PASS_K = 5
DEFAULT_TEMPERATURE = 1.0
DEFAULT_TOP_P = 0.9
DEFAULT_TOP_K = 50
DEFAULT_DO_SAMPLE = True
DEFAULT_SEED = 42
DEFAULT_MM_PROCESSOR_CACHE_GB = 0.0
DEFAULT_JUDGE_WORKERS = 512
DEEPSEEK_JUDGE_THINKING = {"type": "enabled"}
JUDGMENT_METRIC_VERSION = "passk_avgk_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--model-name", default=None)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--datasets", default=",".join(DEFAULT_ZLI_DATASETS))
    parser.add_argument("--benchmarks", default=",".join(DEFAULT_EVAL_BENCHMARKS))
    parser.add_argument("--default-split", default="test")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    parser.add_argument("--top-p", type=float, default=DEFAULT_TOP_P)
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--do-sample", action=argparse.BooleanOptionalAction, default=DEFAULT_DO_SAMPLE)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--pass-k", type=positive_int, default=DEFAULT_PASS_K)
    parser.add_argument("--num-shards", type=positive_int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--pipeline-parallel-size", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--max-model-len", type=int, default=65536)
    parser.add_argument("--max-num-seqs", type=int, default=None)
    parser.add_argument(
        "--disable-custom-all-reduce",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Disable vLLM custom all-reduce for tensor-parallel eval stability.",
    )
    parser.add_argument("--limit-images", type=int, default=16)
    parser.add_argument(
        "--mm-processor-cache-gb",
        type=float,
        default=DEFAULT_MM_PROCESSOR_CACHE_GB,
        help=(
            "vLLM multimodal processor cache size in GiB. Defaults to 0 for "
            "eval stability with large multimodal pass@k batches."
        ),
    )
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--trust-remote-code", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--ignore-errors", action="store_true")
    parser.add_argument("--resume-responses", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--skip-judge", action="store_true")
    parser.add_argument("--judge-only", action="store_true")
    parser.add_argument("--response-file", default=None)
    parser.add_argument("--response-dir", default=None)
    parser.add_argument("--judgment-dir", default=None)
    parser.add_argument("--judge-backend", default="llm", help=argparse.SUPPRESS)
    parser.add_argument("--judge-model", default=DEFAULT_JUDGE_MODEL)
    parser.add_argument("--judge-api-url", default=DEFAULT_JUDGE_API_URL)
    parser.add_argument("--judge-key-env", default=DEFAULT_JUDGE_KEY_ENV)
    parser.add_argument("--judge-proxy", default=DEFAULT_JUDGE_PROXY)
    parser.add_argument("--judge-workers", type=int, default=DEFAULT_JUDGE_WORKERS)
    parser.add_argument("--judge-max-tokens", type=int, default=16384)
    parser.add_argument("--judge-timeout", type=float, default=120.0)
    parser.add_argument("--judge-retries", type=int, default=2)
    parser.add_argument("--judge-log-every", type=int, default=1)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("value must be >= 1")
    return parsed


def validate_shard_args(args: argparse.Namespace) -> None:
    if args.shard_index < 0 or args.shard_index >= args.num_shards:
        raise ValueError(
            "shard-index must satisfy 0 <= shard-index < num-shards; "
            f"got shard-index={args.shard_index}, num-shards={args.num_shards}."
        )


def selected_shard_indices(total: int, num_shards: int, shard_index: int) -> list[int]:
    if num_shards <= 1:
        return list(range(total))
    return list(range(shard_index, total, num_shards))


def normalize_judge_backend(value: str | None) -> str:
    backend = str(value or "llm").strip().lower()
    if backend in {"", "hybrid", "mathruler", "llm", "gemini", "deepseek"}:
        return "llm"
    raise ValueError(
        f"judge backend is fixed to LLM judging, got {value!r}"
    )


def requested_pass_k(args: argparse.Namespace) -> int:
    return max(1, int(getattr(args, "pass_k", 1) or 1))


def effective_question_batch_size(batch_size: int, pass_k: int) -> int:
    return max(1, int(batch_size or 1) // max(1, int(pass_k or 1)))


def judge_schedule(args: argparse.Namespace) -> str:
    if getattr(args, "judge_only", False):
        return "judge_only_synchronous"
    return "async_after_each_response_file"


def judge_key_env_description(args: argparse.Namespace) -> str:
    env_names = []
    for name in (getattr(args, "judge_key_env", ""), "DEEPSEEK_API_KEY", "OPENAI_API_KEY"):
        if name and name not in env_names:
            env_names.append(name)
    return " or ".join(env_names)


def is_deepseek_judge(args: argparse.Namespace) -> bool:
    model = str(getattr(args, "judge_model", "") or "").lower()
    api_url = str(getattr(args, "judge_api_url", "") or "").lower()
    return "deepseek" in model or "deepseek" in api_url


def is_deepseek_v4_flash_judge(args: argparse.Namespace) -> bool:
    model = str(getattr(args, "judge_model", "") or "").lower()
    normalized = model.replace("_", "-").replace(" ", "-")
    return "deepseek" in normalized and "v4" in normalized and "flash" in normalized


def effective_judge_proxy(args: argparse.Namespace) -> str:
    if is_deepseek_judge(args):
        return ""
    return str(getattr(args, "judge_proxy", "") or "")


def judge_thinking_config(args: argparse.Namespace) -> dict[str, str] | None:
    if is_deepseek_v4_flash_judge(args):
        return None
    if is_deepseek_judge(args):
        return dict(DEEPSEEK_JUDGE_THINKING)
    return None


def judge_extra_body(args: argparse.Namespace) -> dict[str, Any] | None:
    thinking = judge_thinking_config(args)
    if thinking is None:
        return None
    return {"thinking": thinking}


def set_eval_seed(seed: int | None) -> None:
    if seed is None:
        return
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except Exception:
        pass
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def main() -> None:
    args = parse_args()
    args.prompt_mode = DEFAULT_EVAL_PROMPT_MODE
    args.judge_backend = normalize_judge_backend(args.judge_backend)
    args.pass_k = requested_pass_k(args)
    validate_shard_args(args)
    set_eval_seed(args.seed)
    output_dir = Path(args.output_dir).resolve()
    if args.dry_run:
        print(json.dumps(config_preview(args, output_dir), indent=2, ensure_ascii=False))
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    summary = initial_summary(args, output_dir)

    if not args.judge_only:
        inference_summary = run_inference_bundle(args, output_dir)
        summary["datasets"] = inference_summary["datasets"]
        summary["benchmark_responses"] = inference_summary["benchmarks"]
        summary["judge"] = inference_summary.get("judge")

    if not args.skip_judge and args.judge_only:
        response_files = collect_response_files(args, output_dir)
        if not response_files:
            raise FileNotFoundError("No response JSONL files were found for judging.")
        summary["judge"] = run_judge(args, output_dir, response_files)
    if not args.skip_judge:
        summary["benchmarks"] = score_benchmarks(args, output_dir)

    summary_path = output_dir / "summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"Summary: {summary_path}")


def initial_summary(args: argparse.Namespace, output_dir: Path) -> dict[str, Any]:
    existing_summary = load_existing_summary(output_dir / "summary.json") if args.judge_only else {}
    return {
        "model_path": str(Path(args.model_path).resolve()),
        "model_name": args.model_name or Path(args.model_path).name,
        "output_dir": str(output_dir),
        "pass_k": requested_pass_k(args),
        "generation": {
            "prompt_mode": args.prompt_mode,
            "prompt_mode_description": prompt_mode_description(args.prompt_mode),
            "assistant_prefill": assistant_prefill_for_prompt_mode(args.prompt_mode),
            "pass_k_generation_mode": "vllm_sampling_params_n",
            "max_tokens": args.max_tokens,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "top_k": args.top_k,
            "do_sample": args.do_sample,
            "seed": args.seed,
            "attempt_seed_policy": "SamplingParams(seed=seed, n=pass_k)",
            "batch_size_semantics": "maximum generated candidates per vLLM call",
            "question_batch_size": effective_question_batch_size(args.batch_size, requested_pass_k(args)),
            "mm_processor_cache_gb": args.mm_processor_cache_gb,
            "num_shards": args.num_shards,
            "shard_index": args.shard_index,
        },
        "datasets": existing_summary.get("datasets", []),
        "benchmark_responses": existing_summary.get("benchmark_responses", []),
        "benchmarks": existing_summary.get("benchmarks", []),
        "judge": existing_summary.get("judge"),
    }


def load_existing_summary(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf-8") as handle:
            summary = json.load(handle)
    except (json.JSONDecodeError, OSError):
        return {}
    return summary if isinstance(summary, dict) else {}


def config_preview(args: argparse.Namespace, output_dir: Path) -> dict[str, Any]:
    return {
        "model_path": str(Path(args.model_path).resolve()),
        "model_name": args.model_name or Path(args.model_path).name,
        "output_dir": str(output_dir),
        "datasets": [spec.__dict__ for spec in parse_dataset_specs(args.datasets, args.default_split)],
        "benchmarks": parse_benchmark_specs(args.benchmarks),
        "limit": args.limit,
        "batch_size": args.batch_size,
        "generation": {
            "prompt_mode": args.prompt_mode,
            "prompt_mode_description": prompt_mode_description(args.prompt_mode),
            "assistant_prefill": assistant_prefill_for_prompt_mode(args.prompt_mode),
            "pass_k_generation_mode": "vllm_sampling_params_n",
            "max_tokens": args.max_tokens,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "top_k": args.top_k,
            "do_sample": args.do_sample,
            "seed": args.seed,
            "attempt_seed_policy": "SamplingParams(seed=seed, n=pass_k)",
            "pass_k": requested_pass_k(args),
            "batch_size_semantics": "maximum generated candidates per vLLM call",
            "question_batch_size": effective_question_batch_size(args.batch_size, requested_pass_k(args)),
            "tensor_parallel_size": args.tensor_parallel_size,
            "pipeline_parallel_size": args.pipeline_parallel_size,
            "gpu_memory_utilization": args.gpu_memory_utilization,
            "max_model_len": args.max_model_len,
            "max_num_seqs": args.max_num_seqs,
            "limit_images": args.limit_images,
            "mm_processor_cache_gb": args.mm_processor_cache_gb,
            "dtype": args.dtype,
            "resume_responses": args.resume_responses,
            "num_shards": args.num_shards,
            "shard_index": args.shard_index,
        },
        "judge": {
            "enabled": not args.skip_judge,
            "judge_only": args.judge_only,
            "backend": normalize_judge_backend(getattr(args, "judge_backend", "llm")),
            "pass_k": requested_pass_k(args),
            "schedule": judge_schedule(args),
            "judgment_record": "compact",
            "model": args.judge_model,
            "api_url": args.judge_api_url,
            "key_env": judge_key_env_description(args),
            "proxy": effective_judge_proxy(args),
            "thinking": judge_thinking_config(args),
            "workers": args.judge_workers,
            "max_tokens": args.judge_max_tokens,
            "timeout": args.judge_timeout,
            "retries": args.judge_retries,
            "log_every": args.judge_log_every,
        },
    }


def run_inference(
    args: argparse.Namespace,
    output_dir: Path,
) -> list[dict[str, Any]]:
    return run_inference_bundle(args, output_dir)["datasets"]


class AsyncJudgeScheduler:
    def __init__(self, args: argparse.Namespace, client: Any):
        self.args = args
        self.client = client
        self.summary = empty_judge_summary(args)
        self.executor = ThreadPoolExecutor(max_workers=1)
        self.jobs: list[dict[str, Any]] = []

    def submit(
        self,
        *,
        label: str,
        response_file: Path,
        judgment_path: Path,
        records: list[dict[str, Any]],
        stats: dict[str, Any],
    ) -> None:
        response_records = list(records)
        print(
            f"{label}: response complete; queued async judge for {response_file.name}",
            flush=True,
        )
        future = self.executor.submit(
            ensure_judgments_for_response_file,
            args=self.args,
            client=self.client,
            response_file=response_file,
            judgment_path=judgment_path,
            records=response_records,
            summary=None,
        )
        self.jobs.append(
            {
                "label": label,
                "response_file": response_file,
                "stats": stats,
                "future": future,
            }
        )

    def wait(self) -> dict[str, Any]:
        try:
            for job in self.jobs:
                file_summary = job["future"].result()
                stats = job["stats"]
                stats["judge_correct"] = file_summary["correct"]
                stats["judge_incorrect"] = file_summary["incorrect"]
                stats["judge_attempt_correct"] = file_summary.get("attempt_correct", 0)
                stats["judge_attempt_total"] = file_summary.get("attempt_total", 0)
                stats["judge_avg_at_k"] = file_summary.get("avg_at_k")
                self.summary["files"].append(file_summary)
                for key in ("total", "correct", "incorrect", "errors"):
                    self.summary[key] += file_summary[key]
                self.summary["attempt_correct"] += file_summary.get("attempt_correct", 0)
                self.summary["attempt_total"] += file_summary.get("attempt_total", 0)
                print(
                    f"Async judged {job['response_file'].name}: "
                    f"{file_summary['correct']}/{file_summary['total']} pass@{file_summary['pass_k']} "
                    f"avg@{file_summary['pass_k']}={format_optional_percent(file_summary.get('avg_at_k'))}",
                    flush=True,
                )
            return finalize_judge_summary(self.summary)
        finally:
            self.executor.shutdown(wait=True)

    def shutdown(self) -> None:
        self.executor.shutdown(wait=False, cancel_futures=True)


def run_inference_bundle(
    args: argparse.Namespace,
    output_dir: Path,
) -> dict[str, Any]:
    from datasets import load_dataset
    from vllm import LLM

    processor = AutoProcessor.from_pretrained(
        args.model_path,
        trust_remote_code=args.trust_remote_code,
        use_fast=False,
    )
    llm = LLM(**vllm_kwargs(args))
    sampling_params = make_sampling_params(
        args,
        max_tokens=args.max_tokens,
        n=requested_pass_k(args),
    )
    judge_scheduler = None
    if not args.skip_judge:
        configure_proxy(effective_judge_proxy(args))
        judge_scheduler = AsyncJudgeScheduler(args, build_openai_client(args))

    response_dir = output_dir / "responses"
    response_dir.mkdir(parents=True, exist_ok=True)
    judgment_dir = Path(args.judgment_dir) if args.judgment_dir else output_dir / "judgments"
    if not args.skip_judge:
        judgment_dir.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, Any]] = []

    try:
        for spec in parse_dataset_specs(args.datasets, args.default_split):
            print(f"Loading dataset {spec.path}@{spec.split}", flush=True)
            dataset = load_dataset(spec.path, split=spec.split)
            total = len(dataset)
            run_total = min(total, args.limit) if args.limit is not None else total
            shard_indices = selected_shard_indices(
                run_total,
                args.num_shards,
                args.shard_index,
            )
            response_path = response_dir / f"{spec.safe_name}.jsonl"
            stats = run_response_file_full_passk(
                args=args,
                llm=llm,
                processor=processor,
                sampling_params=sampling_params,
                label=spec.safe_name,
                source=spec.path,
                total=run_total,
                shard_indices=shard_indices,
                response_path=response_path,
                judgment_path=judgment_dir / response_path.name,
                load_sample=lambda index, dataset=dataset, spec=spec: sample_from_record(
                    spec.path,
                    index,
                    dataset[index],
                ),
                judge_scheduler=judge_scheduler,
            )
            stats["dataset"] = spec.path
            stats["split"] = spec.split
            stats["safe_name"] = spec.safe_name
            results.append(stats)

        benchmark_results = run_benchmark_inference_tasks(
            args=args,
            response_dir=response_dir,
            judgment_dir=judgment_dir,
            processor=processor,
            llm=llm,
            sampling_params=sampling_params,
            judge_scheduler=judge_scheduler,
        )
        judge_summary = judge_scheduler.wait() if judge_scheduler is not None else None
    except Exception:
        if judge_scheduler is not None:
            judge_scheduler.shutdown()
        raise
    return {
        "datasets": results,
        "benchmarks": benchmark_results,
        "judge": judge_summary,
    }


def run_benchmark_inference_tasks(
    *,
    args: argparse.Namespace,
    response_dir: Path,
    judgment_dir: Path,
    processor: Any,
    llm: Any,
    sampling_params: Any,
    judge_scheduler: AsyncJudgeScheduler | None,
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for task in load_benchmark_tasks(getattr(args, "benchmarks", "")):
        print(f"Loading benchmark {task.name} from {task.source}", flush=True)
        total = task.total
        run_total = min(total, args.limit) if args.limit is not None else total
        shard_indices = selected_shard_indices(
            run_total,
            args.num_shards,
            args.shard_index,
        )
        response_path = response_dir / f"{task.response_stem}.jsonl"
        stats = run_response_file_full_passk(
            args=args,
            llm=llm,
            processor=processor,
            sampling_params=sampling_params,
            label=task.response_stem,
            source=f"benchmark/{task.name}",
            total=run_total,
            shard_indices=shard_indices,
            response_path=response_path,
            judgment_path=judgment_dir / response_path.name,
            load_sample=task.load_sample,
            judge_scheduler=judge_scheduler,
        )
        stats["benchmark"] = task.name
        stats["source"] = task.source
        stats["safe_name"] = task.response_stem
        results.append(stats)
    return results


def run_response_file_full_passk(
    *,
    args: argparse.Namespace,
    llm: Any,
    processor: Any,
    sampling_params: Any,
    label: str,
    source: str,
    total: int,
    shard_indices: list[int] | None,
    response_path: Path,
    judgment_path: Path,
    load_sample: Callable[[int], Any],
    judge_scheduler: AsyncJudgeScheduler | None,
) -> dict[str, Any]:
    pass_k = requested_pass_k(args)
    question_batch_size = effective_question_batch_size(args.batch_size, pass_k)
    selected_indices = list(range(total)) if shard_indices is None else list(shard_indices)
    stats = {
        "samples": len(selected_indices),
        "dataset_total_before_shard": total,
        "num_shards": args.num_shards,
        "shard_index": args.shard_index,
        "responses": str(response_path),
        "prompt_mode": args.prompt_mode,
        "prompt_mode_description": prompt_mode_description(args.prompt_mode),
        "assistant_prefill": assistant_prefill_for_prompt_mode(args.prompt_mode),
        "pass_k": pass_k,
        "pass_k_generation_mode": "vllm_sampling_params_n",
        "batch_size": args.batch_size,
        "question_batch_size": question_batch_size,
        "missing_extracted_answer": 0,
        "errors": 0,
        "existing_responses": 0,
        "new_responses": len(selected_indices),
        "generated_attempts": 0,
        "judge_correct": 0,
        "judge_incorrect": 0,
        "judge_attempt_correct": 0,
        "judge_attempt_total": 0,
        "judge_avg_at_k": None,
    }

    records: list[dict[str, Any] | None] = [None] * total
    if args.resume_responses:
        records, response_stats, response_complete = load_resume_response_records(
            response_path,
            total=total,
            required_indices=selected_indices,
            pass_k=pass_k,
            prompt_mode=args.prompt_mode,
        )
        stats.update(response_stats)
        if response_complete:
            existing_records = [
                records[index] for index in selected_indices if records[index] is not None
            ]
            print(
                f"{label}: {len(existing_records)}/{len(selected_indices)} complete "
                f"pass@{pass_k} shard responses; skipping inference",
                flush=True,
            )
            if not args.skip_judge:
                if judge_scheduler is None:
                    raise RuntimeError("Judge scheduler was not initialized.")
                judge_scheduler.submit(
                    label=label,
                    response_file=response_path,
                    judgment_path=judgment_path,
                    records=existing_records,
                    stats=stats,
                )
            return stats
        if stats["existing_responses"]:
            print(
                f"{label}: resuming from {stats['existing_responses']}/"
                f"{len(selected_indices)} existing full pass@{pass_k} shard responses",
                flush=True,
            )

    response_path.parent.mkdir(parents=True, exist_ok=True)
    print(
        f"{label}: generating pass@{pass_k} with vLLM SamplingParams(n={pass_k}); "
        f"question_batch_size={question_batch_size}, "
        f"batch_size_candidate_budget={args.batch_size}, prompt_mode={args.prompt_mode}, "
        f"shard={args.shard_index}/{args.num_shards}, shard_samples={len(selected_indices)}",
        flush=True,
    )
    if not selected_indices:
        write_jsonl(response_path, [])
        if not args.skip_judge:
            write_jsonl(judgment_path, [])
        print(
            f"{label}: shard={args.shard_index}/{args.num_shards} has no samples; "
            "wrote empty response file",
            flush=True,
        )
        return stats

    for start in range(0, len(selected_indices), question_batch_size):
        window_indices = selected_indices[start : start + question_batch_size]
        batch_indices = [index for index in window_indices if records[index] is None]
        if not batch_indices:
            continue
        batch_samples = []
        batch_positions = []
        for index in batch_indices:
            try:
                sample = load_sample(index)
            except Exception as exc:
                if not args.ignore_errors:
                    raise
                record = error_record(
                    source,
                    index,
                    exc,
                    pass_k=pass_k,
                    prompt_mode=args.prompt_mode,
                )
                record["response_index"] = index
                record["pass_k_resolved"] = True
                record["pass_k_stopped_reason"] = "generation_error"
                records[index] = record
                continue
            batch_samples.append(sample)
            batch_positions.append(index)

        if batch_samples:
            generated_records = generate_records(
                args=args,
                llm=llm,
                processor=processor,
                samples=batch_samples,
                sampling_params=sampling_params,
            )
            for index, generated in zip(batch_positions, generated_records):
                generated["response_index"] = index
                generated["pass_k_resolved"] = len(response_attempts(generated)) >= pass_k
                generated["pass_k_stopped_reason"] = (
                    "full_pass_k_generated"
                    if generated["pass_k_resolved"]
                    else "incomplete_generation"
                )
                records[index] = generated
                stats["generated_attempts"] += len(response_attempts(generated))

        concrete_records = [
            records[index] for index in selected_indices if records[index] is not None
        ]
        write_jsonl(response_path, concrete_records)
        generated_so_far = len(concrete_records)
        print(
            f"{label}: generated {generated_so_far}/{len(selected_indices)} shard samples "
            f"({stats['generated_attempts']} new attempts)",
            flush=True,
        )

    concrete_records = [
        records[index] for index in selected_indices if records[index] is not None
    ]
    stats["missing_extracted_answer"] = sum(
        1 for record in concrete_records if not any_attempt_has_answer(record)
    )
    stats["errors"] = sum(1 for record in concrete_records if record.get("generation_error"))

    if not args.skip_judge:
        if judge_scheduler is None:
            raise RuntimeError("Judge scheduler was not initialized.")
        judge_scheduler.submit(
            label=label,
            response_file=response_path,
            judgment_path=judgment_path,
            records=concrete_records,
            stats=stats,
        )
    return stats


def load_resume_response_records(
    response_path: Path,
    *,
    total: int,
    required_indices: list[int] | None = None,
    pass_k: int,
    prompt_mode: str | None = None,
) -> tuple[list[dict[str, Any] | None], dict[str, int], bool]:
    required = set(range(total)) if required_indices is None else set(required_indices)
    records: list[dict[str, Any] | None] = [None] * total
    stats = {
        "existing_responses": 0,
        "new_responses": len(required),
        "missing_extracted_answer": 0,
        "errors": 0,
    }
    if not response_path.exists() or response_path.stat().st_size == 0:
        return records, stats, False

    repair_incomplete_final_response(response_path)
    raw_records = read_jsonl(response_path)
    ignored_records = 0
    for line_index, record in enumerate(raw_records):
        if not isinstance(record, dict):
            ignored_records += 1
            continue
        try:
            response_index = int(record.get("response_index", line_index))
        except (TypeError, ValueError):
            ignored_records += 1
            continue
        if response_index < 0 or response_index >= total:
            ignored_records += 1
            continue
        if response_index not in required:
            ignored_records += 1
            continue
        if not response_record_satisfies_pass_k(record, pass_k, prompt_mode=prompt_mode):
            ignored_records += 1
            continue
        if records[response_index] is not None:
            ignored_records += 1
            continue
        records[response_index] = record

    existing_records = [
        records[index] for index in sorted(required) if records[index] is not None
    ]
    stats["existing_responses"] = len(existing_records)
    stats["new_responses"] = max(len(required) - len(existing_records), 0)
    stats["missing_extracted_answer"] = sum(
        1 for record in existing_records if not any_attempt_has_answer(record)
    )
    stats["errors"] = sum(1 for record in existing_records if record.get("generation_error"))
    if ignored_records:
        print(
            f"{response_path}: ignored {ignored_records} non-resumable response records; "
            "missing records will be regenerated.",
            flush=True,
        )
    return records, stats, len(existing_records) == len(required)


def load_complete_response_records(
    response_path: Path,
    *,
    total: int,
    pass_k: int,
    prompt_mode: str | None = None,
) -> tuple[list[dict[str, Any]] | None, dict[str, int]]:
    stats = {
        "existing_responses": 0,
        "new_responses": total,
        "missing_extracted_answer": 0,
        "errors": 0,
    }
    if not response_path.exists() or response_path.stat().st_size == 0:
        return None, stats
    repair_incomplete_final_response(response_path)
    records = read_jsonl(response_path)
    stats["existing_responses"] = len(records)
    stats["new_responses"] = max(total - len(records), 0)
    stats["missing_extracted_answer"] = sum(
        1 for record in records if not any_attempt_has_answer(record)
    )
    stats["errors"] = sum(1 for record in records if record.get("generation_error"))
    if len(records) != total:
        return None, stats
    if any(
        not response_record_satisfies_pass_k(record, pass_k, prompt_mode=prompt_mode)
        for record in records
    ):
        print(
            f"{response_path}: existing responses are incomplete for full pass@{pass_k}; "
            "or use a different prompt mode; regenerating this response file.",
            flush=True,
        )
        return None, stats
    return records, stats


def judge_records_current(
    *,
    args: argparse.Namespace,
    client: Any,
    records: list[dict[str, Any]],
    response_name: str,
) -> list[dict[str, Any]]:
    if not records:
        return []
    print(
        f"Judging {response_name}: {len(records)} records; workers={args.judge_workers}",
        flush=True,
    )
    if args.judge_workers <= 1:
        return [judge_one(args, client, record) for record in records]

    judgments: list[dict[str, Any] | None] = [None] * len(records)
    with ThreadPoolExecutor(max_workers=args.judge_workers) as pool:
        future_to_index = {
            pool.submit(judge_one, args, client, record): index
            for index, record in enumerate(records)
        }
        for completed, future in enumerate(as_completed(future_to_index), start=1):
            index = future_to_index[future]
            try:
                judgments[index] = future.result()
            except Exception as exc:
                record = records[index]
                judgments[index] = compact_judgment_record(
                    args,
                    record,
                    backend=normalize_judge_backend(getattr(args, "judge_backend", "llm")),
                    verdict="incorrect",
                    reasoning="Judge worker failed.",
                    error=f"{type(exc).__name__}: {exc}",
                )
            if completed % max(1, args.judge_log_every) == 0 or completed == len(records):
                print(f"Judging {response_name}: {completed}/{len(records)} done", flush=True)
    return [judgment for judgment in judgments if judgment is not None]


def write_ordered_judgments(
    judgment_path: Path,
    records: list[dict[str, Any] | None],
    judgments_by_key: dict[str, dict[str, Any]],
) -> None:
    ordered_records = add_response_indices([record for record in records if record is not None])
    ordered_judgments = []
    for record in ordered_records:
        key = judgment_record_key(record)
        if key in judgments_by_key:
            ordered_judgments.append(judgments_by_key[key])
    write_jsonl(judgment_path, ordered_judgments)


def ensure_judgments_for_response_file(
    *,
    args: argparse.Namespace,
    client: Any,
    response_file: Path,
    judgment_path: Path,
    records: list[dict[str, Any]],
    summary: dict[str, Any] | None,
) -> dict[str, Any]:
    indexed_records = add_response_indices(records)
    existing_judgments, done_keys = load_existing_judgments_for_current_policy(
        args,
        response_file=response_file,
        judgment_path=judgment_path,
    )
    newly_judged = judge_records_incremental(
        args=args,
        client=client,
        records=indexed_records,
        output_path=judgment_path,
        done_keys=done_keys,
        response_name=response_file.name,
    )
    judgments = align_judgments_to_records(judgment_path, indexed_records)
    file_summary = summarize_judgments(response_file, judgment_path, judgments)
    file_summary["already_judged"] = len(existing_judgments)
    file_summary["newly_judged"] = newly_judged
    if summary is not None:
        summary["files"].append(file_summary)
        for key in ("total", "correct", "incorrect", "errors"):
            summary[key] += file_summary[key]
        summary["attempt_correct"] += file_summary.get("attempt_correct", 0)
        summary["attempt_total"] += file_summary.get("attempt_total", 0)
    return file_summary


def load_response_resume_state(
    response_path: Path,
    pass_k: int = 1,
    prompt_mode: str | None = None,
) -> tuple[int, dict[str, int]]:
    stats = {
        "missing_extracted_answer": 0,
        "errors": 0,
    }
    if not response_path.exists() or response_path.stat().st_size == 0:
        return 0, stats

    repair_incomplete_final_response(response_path)
    records = read_jsonl(response_path)
    required_pass_k = max(1, int(pass_k or 1))
    if any(
        not response_record_satisfies_pass_k(
            record,
            required_pass_k,
            prompt_mode=prompt_mode,
        )
        for record in records
    ):
        print(
            f"{response_path}: existing responses do not satisfy pass@{required_pass_k}; "
            "or use a different prompt mode; regenerating this response file.",
            flush=True,
        )
        return 0, stats
    stats["missing_extracted_answer"] = sum(
        1 for record in records if not any_attempt_has_answer(record)
    )
    stats["errors"] = sum(1 for record in records if record.get("generation_error"))
    return len(records), stats


def response_record_satisfies_pass_k(
    record: dict[str, Any],
    pass_k: int,
    *,
    prompt_mode: str | None = None,
) -> bool:
    if not response_record_matches_prompt_mode(record, prompt_mode):
        return False
    if record.get("generation_error"):
        return True
    attempts = response_attempts(record)
    required = max(1, int(pass_k or 1))
    return len(attempts) >= required


def response_record_matches_prompt_mode(
    record: dict[str, Any],
    prompt_mode: str | None = None,
) -> bool:
    if prompt_mode is None:
        return True
    expected = normalize_eval_prompt_mode(prompt_mode)
    actual = record.get("prompt_mode")
    if actual:
        try:
            return normalize_eval_prompt_mode(str(actual)) == expected
        except ValueError:
            return False
    if expected != VIGOS_PROMPT_MODE:
        return False
    prompt = str(record.get("prompt") or "")
    response = str(record.get("response") or "")
    vigos_prefill = assistant_prefill_for_prompt_mode(VIGOS_PROMPT_MODE)
    return prompt.endswith(vigos_prefill) or response.startswith(vigos_prefill)


def any_attempt_has_answer(record: dict[str, Any]) -> bool:
    if "judge_has_answer" in record:
        return bool(record["judge_has_answer"])
    return any(
        str(attempt.get("extracted_answer") or "").strip()
        for attempt in response_attempts(record)
    )


def vllm_kwargs(args: argparse.Namespace) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "model": args.model_path,
        "tensor_parallel_size": args.tensor_parallel_size,
        "pipeline_parallel_size": args.pipeline_parallel_size,
        "trust_remote_code": args.trust_remote_code,
        "seed": args.seed,
        "tokenizer_mode": "slow",
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "max_model_len": args.max_model_len,
        "dtype": args.dtype,
        "disable_custom_all_reduce": bool(
            getattr(args, "disable_custom_all_reduce", False)
        ),
        "mm_processor_kwargs": {"use_fast": False},
        # Large pass@k batches repeatedly revisit image prompts.
        # vLLM's mirrored multimodal processor cache can assert if an item is
        # marked cached and then evicted before get_and_update() reaches it.
        # Disabling this cache avoids that unstable path; image preprocessing is
        # recomputed instead of reused.
        "mm_processor_cache_gb": getattr(
            args,
            "mm_processor_cache_gb",
            DEFAULT_MM_PROCESSOR_CACHE_GB,
        ),
    }
    if args.max_num_seqs is not None:
        kwargs["max_num_seqs"] = args.max_num_seqs
    if args.limit_images > 0:
        kwargs["limit_mm_per_prompt"] = {"image": args.limit_images}
    return kwargs


def make_sampling_params(
    args: argparse.Namespace,
    max_tokens: int,
    seed_override: int | None = None,
    n: int = 1,
):
    from vllm import SamplingParams

    kwargs: dict[str, Any] = {
        "max_tokens": max_tokens,
    }
    if int(n or 1) > 1:
        kwargs["n"] = int(n)
    seed = getattr(args, "seed", None) if seed_override is None else seed_override
    if seed is not None:
        kwargs["seed"] = seed
    if getattr(args, "do_sample", DEFAULT_DO_SAMPLE):
        kwargs["temperature"] = args.temperature
        kwargs["top_p"] = args.top_p
        if args.top_k is not None:
            kwargs["top_k"] = args.top_k
    else:
        kwargs["temperature"] = 0.0
    return SamplingParams(**kwargs)


def sampling_param_value(params: Any, name: str, default: Any = None) -> Any:
    if hasattr(params, name):
        return getattr(params, name)
    kwargs = getattr(params, "kwargs", None)
    if isinstance(kwargs, dict) and name in kwargs:
        return kwargs[name]
    return default


def generate_records(
    *,
    args: argparse.Namespace,
    llm: Any,
    processor: Any,
    samples: list[Any],
    sampling_params: Any,
) -> list[dict[str, Any]]:
    prompt_mode = normalize_eval_prompt_mode(getattr(args, "prompt_mode", DEFAULT_EVAL_PROMPT_MODE))
    prompts = [
        build_eval_prompt(
            processor,
            sample.problem,
            sample.images,
            prompt_mode=prompt_mode,
        )
        for sample in samples
    ]
    requests = [vllm_request(prompt, sample.images) for prompt, sample in zip(prompts, samples)]
    outputs = llm.generate(requests, sampling_params)
    sampling_seed = sampling_param_value(
        sampling_params,
        "seed",
        getattr(args, "seed", None),
    )

    records = []
    for sample, prompt, output in zip(samples, prompts, outputs):
        attempts = []
        for attempt_index, candidate in enumerate(
            output.outputs[: requested_pass_k(args)]
        ):
            completion = candidate.text
            response = response_from_completion(completion, prompt_mode=prompt_mode)
            attempts.append(
                {
                    "attempt_index": attempt_index,
                    "response": response,
                    "raw_completion": completion,
                    "extracted_answer": extract_model_answer(response),
                    "sampling_seed": sampling_seed,
                }
            )
        first_attempt = attempts[0] if attempts else {}
        record = {
            "dataset": sample.dataset,
            "sample_id": sample.sample_id,
            "problem": sample.problem,
            "prompt": prompt,
            "prompt_mode": prompt_mode,
            "prompt_mode_description": prompt_mode_description(prompt_mode),
            "assistant_prefill": assistant_prefill_for_prompt_mode(prompt_mode),
            "response": first_attempt.get("response", ""),
            "raw_completion": first_attempt.get("raw_completion", ""),
            "extracted_answer": first_attempt.get("extracted_answer", ""),
            "attempts": attempts,
            "pass_k_requested": requested_pass_k(args),
            "pass_k_returned": len(attempts),
            "seed": getattr(args, "seed", None),
            "attempt_seed_policy": "SamplingParams(seed=seed, n=pass_k)",
            "pass_k_generation_mode": "vllm_sampling_params_n",
            "ground_truth": sample.ground_truth,
            "image_metadata": sample.image_metadata,
            "generation_error": None,
        }
        benchmark_meta = sample.raw.get("benchmark_meta") if isinstance(sample.raw, dict) else None
        if benchmark_meta is not None:
            record["benchmark_meta"] = benchmark_meta
        records.append(record)
    return records


def error_record(
    dataset: str,
    index: int,
    exc: Exception,
    pass_k: int | None = None,
    prompt_mode: str | None = None,
) -> dict[str, Any]:
    normalized_prompt_mode = normalize_eval_prompt_mode(prompt_mode)
    return {
        "dataset": dataset,
        "sample_id": str(index),
        "prompt": "",
        "prompt_mode": normalized_prompt_mode,
        "prompt_mode_description": prompt_mode_description(normalized_prompt_mode),
        "assistant_prefill": assistant_prefill_for_prompt_mode(normalized_prompt_mode),
        "response": "",
        "raw_completion": "",
        "extracted_answer": "",
        "attempts": [],
        "pass_k_requested": pass_k,
        "pass_k_returned": 0,
        "ground_truth": "",
        "image_metadata": [],
        "generation_error": f"{type(exc).__name__}: {exc}",
    }


def response_attempts(record: dict[str, Any]) -> list[dict[str, Any]]:
    attempts = record.get("attempts")
    if isinstance(attempts, list):
        normalized = []
        for index, attempt in enumerate(attempts):
            if not isinstance(attempt, dict):
                continue
            normalized.append(
                {
                    "attempt_index": attempt.get("attempt_index", index),
                    "response": str(attempt.get("response") or ""),
                    "raw_completion": str(attempt.get("raw_completion") or ""),
                    "extracted_answer": str(attempt.get("extracted_answer") or ""),
                    "sampling_seed": attempt.get("sampling_seed"),
                }
            )
        return normalized
    judge_answers = record.get("judge_extracted_answers")
    if isinstance(judge_answers, list):
        return [
            {
                "attempt_index": index,
                "response": "",
                "raw_completion": "",
                "extracted_answer": str(answer or ""),
                "sampling_seed": None,
            }
            for index, answer in enumerate(judge_answers)
        ]
    return [
        {
            "attempt_index": 0,
            "response": str(record.get("response") or ""),
            "raw_completion": str(record.get("raw_completion") or ""),
            "extracted_answer": str(record.get("extracted_answer") or ""),
            "sampling_seed": record.get("sampling_seed", record.get("seed")),
        }
    ]


def normalize_attempt_verdicts(values: Any) -> list[str]:
    if not isinstance(values, list):
        return []
    verdicts = []
    for value in values:
        if isinstance(value, bool):
            verdicts.append("correct" if value else "incorrect")
            continue
        normalized = str(value).strip().lower()
        verdicts.append("correct" if normalized in {"correct", "true", "yes", "1"} else "incorrect")
    return verdicts


def attempt_verdicts_from_judgment(judgment: dict[str, Any]) -> list[str]:
    return normalize_attempt_verdicts(judgment.get("judge_attempt_verdicts"))


def count_correct_attempts(verdicts: list[str]) -> int:
    return sum(1 for verdict in verdicts if verdict == "correct")


def format_optional_percent(value: Any) -> str:
    if value is None:
        return "n/a"
    return f"{float(value) * 100:.2f}%"


def collect_response_files(args: argparse.Namespace, output_dir: Path) -> list[Path]:
    files: list[Path] = []
    if args.response_file:
        files.append(Path(args.response_file))
    response_dir = Path(args.response_dir) if args.response_dir else output_dir / "responses"
    if response_dir.exists():
        if args.response_dir:
            files.extend(sorted(response_dir.rglob("*.jsonl")))
        else:
            expected_stems = [
                spec.safe_name
                for spec in parse_dataset_specs(args.datasets, args.default_split)
            ]
            expected_stems.extend(
                benchmark_response_stem(benchmark)
                for benchmark in parse_benchmark_specs(args.benchmarks)
            )
            files.extend(
                response_dir / f"{stem}.jsonl"
                for stem in expected_stems
                if (response_dir / f"{stem}.jsonl").exists()
            )
    unique = []
    seen: set[Path] = set()
    for file in files:
        resolved = file.resolve()
        if resolved not in seen:
            unique.append(resolved)
            seen.add(resolved)
    return unique


def empty_judge_summary(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "backend": normalize_judge_backend(getattr(args, "judge_backend", "llm")),
        "pass_k": requested_pass_k(args),
        "schedule": judge_schedule(args),
        "judgment_record": "compact",
        "model": args.judge_model,
        "api_url": args.judge_api_url,
        "proxy": effective_judge_proxy(args),
        "thinking": judge_thinking_config(args),
        "record_workers": args.judge_workers,
        "files": [],
        "total": 0,
        "correct": 0,
        "incorrect": 0,
        "errors": 0,
        "attempt_correct": 0,
        "attempt_total": 0,
    }


def finalize_judge_summary(summary: dict[str, Any]) -> dict[str, Any]:
    pass_at_k = summary["correct"] / summary["total"] if summary["total"] else 0.0
    summary["accuracy"] = pass_at_k
    summary["pass_at_k"] = pass_at_k
    summary["avg_at_k"] = (
        summary["attempt_correct"] / summary["attempt_total"]
        if summary.get("attempt_total")
        else None
    )
    return summary


def run_judge(
    args: argparse.Namespace,
    output_dir: Path,
    response_files: list[Path],
) -> dict[str, Any]:
    configure_proxy(effective_judge_proxy(args))
    client = build_openai_client(args)
    judgment_dir = Path(args.judgment_dir) if args.judgment_dir else output_dir / "judgments"
    judgment_dir.mkdir(parents=True, exist_ok=True)

    summary = empty_judge_summary(args)
    for response_file in response_files:
        records = add_response_indices(read_jsonl(response_file))
        output_path = judgment_dir / response_file.name
        existing_judgments, done_keys = load_existing_judgments_for_current_policy(
            args,
            response_file=response_file,
            judgment_path=output_path,
        )
        newly_judged = judge_records_incremental(
            args=args,
            client=client,
            records=records,
            output_path=output_path,
            done_keys=done_keys,
            response_name=response_file.name,
        )
        judgments = align_judgments_to_records(output_path, records)
        file_summary = summarize_judgments(response_file, output_path, judgments)
        file_summary["already_judged"] = len(existing_judgments)
        file_summary["newly_judged"] = newly_judged
        summary["files"].append(file_summary)
        summary["total"] += file_summary["total"]
        summary["correct"] += file_summary["correct"]
        summary["incorrect"] += file_summary["incorrect"]
        summary["errors"] += file_summary["errors"]
        summary["attempt_correct"] += file_summary.get("attempt_correct", 0)
        summary["attempt_total"] += file_summary.get("attempt_total", 0)
        print(
            f"Judged {response_file.name}: "
            f"{file_summary['correct']}/{file_summary['total']} pass@{file_summary['pass_k']} "
            f"avg@{file_summary['pass_k']}={format_optional_percent(file_summary.get('avg_at_k'))} "
            f"({newly_judged} new)",
            flush=True,
        )
    return finalize_judge_summary(summary)


def score_benchmarks(args: argparse.Namespace, output_dir: Path) -> list[dict[str, Any]]:
    response_dir = Path(args.response_dir) if args.response_dir else output_dir / "responses"
    judgment_dir = Path(args.judgment_dir) if args.judgment_dir else output_dir / "judgments"
    results = []
    for benchmark in parse_benchmark_specs(getattr(args, "benchmarks", "")):
        stem = benchmark_response_stem(benchmark)
        response_file = response_dir / f"{stem}.jsonl"
        judgment_file = judgment_dir / f"{stem}.jsonl"
        if not response_file.exists():
            results.append(
                {
                    "benchmark": benchmark,
                    "skipped": True,
                    "reason": f"Response file not found: {response_file}",
                }
            )
            continue
        if not judgment_file.exists():
            results.append(
                {
                    "benchmark": benchmark,
                    "skipped": True,
                    "reason": f"Judgment file not found: {judgment_file}",
                    "response_file": str(response_file),
                }
            )
            continue
        results.append(score_benchmark(benchmark, response_file, judgment_file, output_dir))
    return results


def configure_proxy(proxy: str | None) -> None:
    if proxy:
        for key in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"):
            os.environ[key] = proxy
    no_proxy = os.environ.get("NO_PROXY") or os.environ.get("no_proxy") or ""
    entries = [entry for entry in no_proxy.split(",") if entry]
    for host in ("127.0.0.1", "localhost"):
        if host not in entries:
            entries.append(host)
    value = ",".join(entries)
    os.environ["NO_PROXY"] = value
    os.environ["no_proxy"] = value


def build_openai_client(args: argparse.Namespace):
    import httpx
    from openai import OpenAI

    env_names = []
    for name in (getattr(args, "judge_key_env", ""), "DEEPSEEK_API_KEY", "OPENAI_API_KEY"):
        if name and name not in env_names:
            env_names.append(name)
    api_key = next((os.environ[name] for name in env_names if os.environ.get(name)), "")
    if not api_key:
        raise EnvironmentError(f"{' or '.join(env_names)} is required.")
    http_client = None
    limits = httpx.Limits(
        max_connections=max(args.judge_workers * 2, 20),
        max_keepalive_connections=max(args.judge_workers, 20),
    )
    proxy = effective_judge_proxy(args)
    if proxy:
        http_client = httpx.Client(
            proxy=proxy,
            timeout=args.judge_timeout,
            limits=limits,
        )
    elif is_deepseek_judge(args):
        http_client = httpx.Client(
            timeout=args.judge_timeout,
            limits=limits,
            trust_env=False,
        )
    return OpenAI(
        api_key=api_key,
        base_url=args.judge_api_url,
        timeout=args.judge_timeout,
        http_client=http_client,
    )


def repair_incomplete_final_judgment(path: Path) -> None:
    repair_incomplete_final_jsonl_line(path, "judgment")


def repair_incomplete_final_response(path: Path) -> None:
    repair_incomplete_final_jsonl_line(path, "response")


def repair_incomplete_final_jsonl_line(path: Path, label: str) -> None:
    if not path.exists() or path.stat().st_size == 0:
        return
    text = path.read_text(encoding="utf-8")
    if text.endswith("\n"):
        return
    final_line_start = text.rfind("\n") + 1
    final_line = text[final_line_start:]
    if not final_line.strip():
        path.write_text(text[:final_line_start], encoding="utf-8")
        return
    try:
        json.loads(final_line)
    except json.JSONDecodeError:
        backup_path = path.with_name(f"{path.name}.incomplete-{int(time.time())}")
        backup_path.write_text(final_line, encoding="utf-8")
        path.write_text(text[:final_line_start], encoding="utf-8")
        print(
            f"Recovered incomplete final {label} line from {path}; "
            f"saved fragment to {backup_path}",
            file=sys.stderr,
            flush=True,
        )
    else:
        path.write_text(text + "\n", encoding="utf-8")


def load_existing_judgments(path: Path) -> tuple[list[dict[str, Any]], set[str]]:
    if not path.exists():
        return [], set()

    judgments: list[dict[str, Any]] = []
    done_keys: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                judgment = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid judgment JSONL at {path}:{line_number}") from exc
            key = judgment_record_key(judgment)
            if key in done_keys:
                continue
            judgments.append(judgment)
            done_keys.add(key)
    return judgments, done_keys


def load_existing_judgments_for_current_policy(
    args: argparse.Namespace,
    *,
    response_file: Path,
    judgment_path: Path,
) -> tuple[list[dict[str, Any]], set[str]]:
    repair_incomplete_final_judgment(judgment_path)
    existing_judgments, done_keys = load_existing_judgments(judgment_path)
    if not existing_judgments:
        return existing_judgments, done_keys

    stale = (
        response_file.exists()
        and judgment_path.exists()
        and judgment_path.stat().st_mtime < response_file.stat().st_mtime
    )
    invalid_policy = any(
        not judgment_record_satisfies_current_policy(args, judgment)
        for judgment in existing_judgments
    )
    if not stale and not invalid_policy:
        return existing_judgments, done_keys

    reason = (
        "stale response file"
        if stale
        else "non-LLM, non-compact, incomplete pass@k/avg@k, or mismatched judge thinking policy"
    )
    backup_path = judgment_path.with_name(f"{judgment_path.name}.legacy-{int(time.time())}")
    judgment_path.replace(backup_path)
    print(
        f"Ignoring existing judgments for {judgment_path.name} due to {reason}; "
        f"backed up to {backup_path}",
        flush=True,
    )
    return [], set()


def judgment_record_satisfies_current_policy(
    args: argparse.Namespace,
    judgment: dict[str, Any],
) -> bool:
    if str(judgment.get("judgment_record") or "").strip().lower() != "compact":
        return False
    if str(judgment.get("judge_backend") or "").strip().lower() != "llm":
        return False
    pass_k = requested_pass_k(args)
    try:
        judged_pass_k = int(judgment.get("judge_pass_k") or judgment.get("pass_k_requested") or 1)
    except (TypeError, ValueError):
        return False
    if judged_pass_k != pass_k:
        return False
    expected_thinking = judge_thinking_config(args)
    if expected_thinking is not None and judgment.get("judge_thinking") != expected_thinking:
        return False
    if expected_thinking is None and is_deepseek_v4_flash_judge(args) and "judge_thinking" in judgment:
        return False
    if judgment.get("generation_error"):
        return True
    if judgment.get("judgment_metric_version") != JUDGMENT_METRIC_VERSION:
        return False
    try:
        attempt_count = int(judgment.get("judge_attempt_count") or 0)
    except (TypeError, ValueError):
        return False
    attempt_verdicts = attempt_verdicts_from_judgment(judgment)
    return attempt_count >= pass_k and len(attempt_verdicts) == attempt_count


def judgment_record_key(record: dict[str, Any]) -> str:
    dataset = str(record.get("dataset") or "")
    sample_id = str(record.get("sample_id") or "")
    pass_k = record.get("pass_k_requested") or record.get("judge_pass_k")
    pass_k_suffix = f"\tpass@{pass_k}" if pass_k else ""
    if "response_index" in record:
        return f"{dataset}\t{sample_id}\t{record['response_index']}{pass_k_suffix}"
    if sample_id:
        return f"{dataset}\t{sample_id}{pass_k_suffix}"
    return json.dumps(
        {
            "dataset": dataset,
            "prompt": record.get("prompt") or "",
            "ground_truth": record.get("ground_truth") or "",
            "pass_k": pass_k,
        },
        sort_keys=True,
        ensure_ascii=False,
    )


def add_response_indices(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {**record, "response_index": record.get("response_index", index)}
        for index, record in enumerate(records)
    ]


def pending_judge_records(
    records: list[dict[str, Any]],
    done_keys: set[str],
) -> list[dict[str, Any]]:
    pending = []
    seen_pending: set[str] = set()
    for record in records:
        key = judgment_record_key(record)
        if key in done_keys or key in seen_pending:
            continue
        pending.append(record)
        seen_pending.add(key)
    return pending


def align_judgments_to_records(
    judgment_path: Path,
    records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    judgments, _ = load_existing_judgments(judgment_path)
    judgment_by_key = {judgment_record_key(judgment): judgment for judgment in judgments}
    return [
        judgment_by_key[key]
        for key in (judgment_record_key(record) for record in records)
        if key in judgment_by_key
    ]


def judge_records_incremental(
    *,
    args: argparse.Namespace,
    client: Any,
    records: list[dict[str, Any]],
    output_path: Path,
    done_keys: set[str],
    response_name: str,
) -> int:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pending = pending_judge_records(records, done_keys)
    record_keys = {judgment_record_key(record) for record in records}
    total = len(record_keys)
    completed = len(record_keys & done_keys)
    if not pending:
        print(
            f"Judging {response_name}: {completed}/{total} already done; skipping API calls.",
            flush=True,
        )
        return 0

    print(
        f"Judging {response_name}: {completed}/{total} already done; "
        f"{len(pending)} pending; workers={args.judge_workers}",
        flush=True,
    )
    newly_judged = 0
    log_every = max(1, args.judge_log_every)

    def write_judgment(writer: Any, judgment: dict[str, Any]) -> None:
        nonlocal completed, newly_judged
        key = judgment_record_key(judgment)
        if key in done_keys:
            return
        writer.write(json.dumps(judgment, ensure_ascii=False) + "\n")
        writer.flush()
        done_keys.add(key)
        completed += 1
        newly_judged += 1
        if newly_judged % log_every == 0 or completed == total:
            print(f"Judging {response_name}: {completed}/{total} done", flush=True)

    with output_path.open("a", encoding="utf-8") as writer:
        if args.judge_workers <= 1:
            for record in pending:
                write_judgment(writer, judge_one(args, client, record))
            return newly_judged

        with ThreadPoolExecutor(max_workers=args.judge_workers) as pool:
            future_to_record = {
                pool.submit(judge_one, args, client, record): record for record in pending
            }
            for future in as_completed(future_to_record):
                record = future_to_record[future]
                try:
                    judgment = future.result()
                except Exception as exc:
                    judgment = compact_judgment_record(
                        args,
                        record,
                        backend=normalize_judge_backend(getattr(args, "judge_backend", "llm")),
                        verdict="incorrect",
                        reasoning="Judge worker failed.",
                        error=f"{type(exc).__name__}: {exc}",
                    )
                write_judgment(writer, judgment)
    return newly_judged


def judge_one(args: argparse.Namespace, client: Any, record: dict[str, Any]) -> dict[str, Any]:
    normalize_judge_backend(getattr(args, "judge_backend", "llm"))
    return judge_one_llm(args, client, record, backend="llm")


def compact_judgment_record(
    args: argparse.Namespace,
    record: dict[str, Any],
    *,
    backend: str,
    verdict: str,
    reasoning: str,
    error: str | None,
    attempts: list[dict[str, Any]] | None = None,
    extracted_answers: list[str] | None = None,
    attempt_verdicts: list[str] | None = None,
    success_attempt: int | None = None,
) -> dict[str, Any]:
    pass_k = requested_pass_k(args)
    attempts = attempts if attempts is not None else response_attempts(record)[:pass_k]
    if extracted_answers is None:
        extracted_answers = [str(attempt.get("extracted_answer") or "") for attempt in attempts]
    if attempt_verdicts is None:
        if len(attempts) <= 1:
            attempt_verdicts = [verdict] if attempts else []
        elif error:
            attempt_verdicts = ["incorrect"] * len(attempts)
        else:
            attempt_verdicts = []
    attempt_verdicts = normalize_attempt_verdicts(attempt_verdicts)
    attempt_correct_count = count_correct_attempts(attempt_verdicts)
    if success_attempt is None and attempt_correct_count:
        for attempt, attempt_verdict in zip(attempts, attempt_verdicts):
            if attempt_verdict == "correct":
                success_attempt = int(attempt.get("attempt_index", 0) or 0)
                break

    judgment: dict[str, Any] = {
        "dataset": str(record.get("dataset") or ""),
        "sample_id": str(record.get("sample_id") or ""),
        "judgment_record": "compact",
        "judgment_metric_version": JUDGMENT_METRIC_VERSION,
        "pass_k_requested": record.get("pass_k_requested", pass_k),
        "judge_backend": backend,
        "judge_pass_k": pass_k,
        "judge_attempt_count": len(attempts),
        "judge_has_answer": any(answer.strip() for answer in extracted_answers),
        "judge_extracted_answers": extracted_answers,
        "judge_attempt_verdicts": attempt_verdicts,
        "judge_attempt_correct_count": attempt_correct_count,
        "avg_at_k": attempt_correct_count / len(attempt_verdicts) if attempt_verdicts else None,
        "judge_success_attempt": success_attempt,
        "judge_verdict": verdict,
        "judge_reasoning": reasoning,
        "judge_error": error,
    }
    if "response_index" in record:
        judgment["response_index"] = record["response_index"]
    thinking = judge_thinking_config(args)
    if thinking is not None:
        judgment["judge_thinking"] = thinking
    benchmark_meta = record.get("benchmark_meta")
    if benchmark_meta is not None:
        judgment["benchmark_meta"] = benchmark_meta
    generation_error = record.get("generation_error")
    if generation_error:
        judgment["generation_error"] = generation_error
    return judgment


def judge_one_llm(
    args: argparse.Namespace,
    client: Any,
    record: dict[str, Any],
    *,
    attempts: list[dict[str, Any]] | None = None,
    backend: str = "llm",
) -> dict[str, Any]:
    ground_truth = str(record.get("ground_truth") or "")
    attempts = attempts if attempts is not None else response_attempts(record)[: requested_pass_k(args)]
    extracted_answers = [str(attempt.get("extracted_answer") or "") for attempt in attempts]
    extracted_answer = extracted_answers[0] if extracted_answers else str(record.get("extracted_answer") or "")
    if len(extracted_answers) <= 1:
        messages = build_judge_messages(
            extracted_answer,
            ground_truth,
            question_context=judge_question_context(record),
        )
    else:
        messages = build_passk_judge_messages(
            extracted_answers,
            ground_truth,
            question_context=judge_question_context(record),
        )
    last_error = ""
    for attempt in range(args.judge_retries + 1):
        try:
            request_kwargs = {
                "model": args.judge_model,
                "messages": messages,
                "temperature": 0.0,
                "max_tokens": args.judge_max_tokens,
            }
            extra_body = judge_extra_body(args)
            if extra_body is not None:
                request_kwargs["extra_body"] = extra_body
            response = client.chat.completions.create(**request_kwargs)
            judge_text = response.choices[0].message.content or ""
            parsed = parse_judge_output(judge_text)
            attempt_verdicts = normalize_attempt_verdicts(parsed.get("attempt_verdicts"))
            if len(attempts) > 1:
                if len(attempt_verdicts) != len(attempts):
                    raise ValueError(
                        "Judge output missing attempt_verdicts with one verdict per attempt."
                    )
                verdict = "correct" if count_correct_attempts(attempt_verdicts) else "incorrect"
            else:
                verdict = parsed["verdict"]
                attempt_verdicts = [verdict] if attempts else []
            return compact_judgment_record(
                args,
                record,
                backend=backend,
                verdict=verdict,
                reasoning=parsed["reasoning"],
                error=None,
                attempts=attempts,
                extracted_answers=extracted_answers,
                attempt_verdicts=attempt_verdicts,
            )
        except Exception as exc:
            last_error = f"{type(exc).__name__}: {exc}"
            if attempt < args.judge_retries:
                time.sleep(min(2**attempt, 8))
    return compact_judgment_record(
        args,
        record,
        backend=backend,
        verdict="incorrect",
        reasoning="Judge request failed.",
        error=last_error,
        attempts=attempts,
        extracted_answers=extracted_answers,
    )


def judge_question_context(record: dict[str, Any]) -> str:
    problem = str(record.get("problem") or "").strip()
    if problem:
        return problem

    prompt = str(record.get("prompt") or "").strip()
    if not prompt:
        return ""
    if "Problem:" not in prompt:
        return prompt

    context = "Problem:" + prompt.split("Problem:", 1)[1]
    for marker in (
        "\n\nYou are tasked with analyzing",
        "\nYou are tasked with analyzing",
        "<|im_end|>",
    ):
        if marker in context:
            context = context.split(marker, 1)[0]
    return context.strip()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def summarize_judgments(
    response_file: Path,
    judgment_file: Path,
    judgments: list[dict[str, Any]],
) -> dict[str, Any]:
    total = len(judgments)
    correct = sum(1 for item in judgments if item.get("judge_verdict") == "correct")
    errors = sum(1 for item in judgments if item.get("judge_error"))
    missing_answer = sum(1 for item in judgments if not any_attempt_has_answer(item))
    attempt_counts = [
        int(item.get("judge_attempt_count", len(response_attempts(item))) or 0)
        for item in judgments
    ]
    judge_backends = sorted(
        {
            str(item.get("judge_backend") or "").strip()
            for item in judgments
            if item.get("judge_backend")
        }
    )
    pass_ks = [
        int(item.get("judge_pass_k") or item.get("pass_k_requested") or 1)
        for item in judgments
    ]
    attempt_verdicts = [attempt_verdicts_from_judgment(item) for item in judgments]
    attempt_total = sum(len(verdicts) for verdicts in attempt_verdicts)
    attempt_correct = sum(count_correct_attempts(verdicts) for verdicts in attempt_verdicts)
    pass_at_k = correct / total if total else 0.0
    return {
        "response_file": str(response_file),
        "judgment_file": str(judgment_file),
        "dataset": sanitize_dataset_name(response_file.stem),
        "judge_backend": judge_backends[0] if len(judge_backends) == 1 else judge_backends,
        "pass_k": max(pass_ks) if pass_ks else 1,
        "mean_attempts": sum(attempt_counts) / len(attempt_counts) if attempt_counts else 0.0,
        "total": total,
        "correct": correct,
        "incorrect": total - correct,
        "errors": errors,
        "missing_extracted_answer": missing_answer,
        "accuracy": pass_at_k,
        "pass_at_k": pass_at_k,
        "attempt_correct": attempt_correct,
        "attempt_total": attempt_total,
        "avg_at_k": attempt_correct / attempt_total if attempt_total else None,
    }


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        raise
    except Exception as exc:
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        raise
