#!/usr/bin/env python
"""Merge a PEFT LoRA adapter into a Qwen2.5-VL or Qwen3-VL base checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoConfig, AutoProcessor

try:
    from transformers import Qwen2_5_VLForConditionalGeneration
    from transformers import Qwen3VLForConditionalGeneration
except ImportError as exc:  # pragma: no cover - depends on installed transformers version.
    raise ImportError(
        "Qwen2.5-VL or Qwen3-VL model classes are unavailable. Run "
        "`uv sync --python 3.11` with the pinned ViGOS environment first."
    ) from exc


DTYPES = {
    "bfloat16": torch.bfloat16,
    "bf16": torch.bfloat16,
    "float16": torch.float16,
    "fp16": torch.float16,
    "float32": torch.float32,
    "fp32": torch.float32,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge a ViGOS LoRA adapter into its Qwen2.5-VL or Qwen3-VL base model."
    )
    parser.add_argument(
        "--base-model",
        default=None,
        help=(
            "Path or HF id of the base checkpoint. If omitted, the script reads "
            "base_model_name_or_path from the adapter's adapter_config.json."
        ),
    )
    parser.add_argument(
        "--adapter",
        default="runs/vigos_qwen25_3b_YYYYMMDD-HHMMSS/checkpoint-500",
        help="Path to the LoRA adapter directory or checkpoint-* directory.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help=(
            "Directory for the merged full checkpoint. If omitted, writes to "
            "<adapter-dir>_merged next to the adapter directory."
        ),
    )
    parser.add_argument(
        "--dtype",
        default="bfloat16",
        choices=sorted(DTYPES),
        help="Weight dtype used while loading and saving the merged model.",
    )
    parser.add_argument(
        "--device-map",
        default="cpu",
        help='Transformers device_map for loading. Use "cpu" by default, or "auto".',
    )
    parser.add_argument(
        "--attn-implementation",
        default="sdpa",
        help="Attention implementation used only for model loading; sdpa is safest on CPU.",
    )
    parser.add_argument(
        "--trust-remote-code",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Pass trust_remote_code to Transformers loaders.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow writing into an existing non-empty output directory.",
    )
    return parser.parse_args()


def require_adapter_files(adapter_dir: Path) -> None:
    missing = ["adapter_config.json"] if not (adapter_dir / "adapter_config.json").is_file() else []
    has_adapter_weights = any(
        (adapter_dir / name).is_file()
        for name in ("adapter_model.safetensors", "adapter_model.bin")
    )
    if not has_adapter_weights:
        missing.append("adapter_model.safetensors or adapter_model.bin")
    if missing:
        raise FileNotFoundError(
            f"{adapter_dir} is not a complete LoRA adapter directory; missing "
            f"{', '.join(missing)}."
        )


def validate_output_dir(output_dir: Path, overwrite: bool) -> None:
    if not output_dir.exists():
        return
    if not output_dir.is_dir():
        raise NotADirectoryError(f"Output path exists but is not a directory: {output_dir}")
    if any(output_dir.iterdir()) and not overwrite:
        raise FileExistsError(
            f"Output directory is not empty: {output_dir}. "
            "Pass --overwrite or choose a fresh output directory."
        )


def load_processor(base_model: str, adapter_dir: Path, trust_remote_code: bool):
    try:
        return AutoProcessor.from_pretrained(
            adapter_dir,
            trust_remote_code=trust_remote_code,
            use_fast=False,
        )
    except Exception:
        return AutoProcessor.from_pretrained(
            base_model,
            trust_remote_code=trust_remote_code,
            use_fast=False,
        )


def read_adapter_base_model(adapter_dir: Path) -> str:
    config_path = adapter_dir / "adapter_config.json"
    with config_path.open("r", encoding="utf-8") as handle:
        adapter_config = json.load(handle)
    base_model = adapter_config.get("base_model_name_or_path")
    if not base_model:
        raise ValueError(
            f"{config_path} does not contain base_model_name_or_path. "
            "Pass --base-model explicitly."
        )
    return str(base_model)


def default_output_dir(adapter_dir: Path) -> Path:
    return adapter_dir.parent / f"{adapter_dir.name}_merged"


def model_class_for_checkpoint(model_name_or_path: str, trust_remote_code: bool = True):
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
        "Unsupported merge base model_type. Expected one of "
        "{'qwen2_5_vl', 'qwen3_vl'}, "
        f"got {model_type!r} from {model_name_or_path!r}."
    )


def main() -> None:
    args = parse_args()
    adapter_dir = Path(args.adapter)
    require_adapter_files(adapter_dir)
    base_model_name_or_path = args.base_model or read_adapter_base_model(adapter_dir)
    output_dir = Path(args.output) if args.output else default_output_dir(adapter_dir)

    validate_output_dir(output_dir, args.overwrite)
    output_dir.mkdir(parents=True, exist_ok=True)

    load_kwargs = {
        "dtype": DTYPES[args.dtype],
        "attn_implementation": args.attn_implementation,
        "trust_remote_code": args.trust_remote_code,
        "low_cpu_mem_usage": True,
    }
    if args.device_map and args.device_map.lower() != "none":
        load_kwargs["device_map"] = args.device_map

    model_class, model_type = model_class_for_checkpoint(
        base_model_name_or_path,
        trust_remote_code=args.trust_remote_code,
    )
    print(
        f"Loading base model: {base_model_name_or_path} "
        f"(model_type={model_type}, class={model_class.__name__})"
    )
    base_model = model_class.from_pretrained(
        base_model_name_or_path,
        **load_kwargs,
    )
    base_model.config.use_cache = True

    print(f"Loading LoRA adapter: {adapter_dir}")
    model = PeftModel.from_pretrained(base_model, adapter_dir)

    print("Merging adapter into base weights")
    model = model.merge_and_unload()

    print(f"Saving merged model: {output_dir}")
    model.save_pretrained(output_dir, safe_serialization=True)

    processor = load_processor(base_model_name_or_path, adapter_dir, args.trust_remote_code)
    processor.save_pretrained(output_dir)

    print("Done.")


if __name__ == "__main__":
    main()
