"""Shared helpers for ViGOS evaluation."""

from __future__ import annotations

import ast
import json
import re
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any

from PIL import Image

from .answer_utils import extract_boxed_content, normalize_reference_answer
from .data_collator import (
    DESCRIPTION_PREFILL,
    format_vigos_student_prompt,
)

VIGOS_PROMPT_MODE = "vigos"
DEFAULT_EVAL_PROMPT_MODE = VIGOS_PROMPT_MODE

DEFAULT_ZLI_DATASETS = [
    "zli12321/mmstar",
    "zli12321/mm-vet",
    "zli12321/visnumbench",
    "zli12321/mmmu_pro_10options",
    "zli12321/mmmu-pro-vision",
    "zli12321/hallusionbench",
    "zli12321/MMMU",
    "zli12321/MMSI",
    "zli12321/mathverse",
    "zli12321/mathvision",
    "zli12321/mathvista",
    "zli12321/realWorldQA",
]

JUDGE_PROMPT_TEMPLATE = """\
You are an expert answer evaluator. Determine whether the model's extracted answer is correct by comparing it to the ground truth.

## Question and Options:
{question_context}

## Model's Extracted Answer:
{extracted_answer}

## Ground Truth Answer:
{ground_truth}

## Instructions:
Compare the model answer to the ground truth in the context of the question and options.
Be flexible with formatting differences (e.g. "1/2" vs "0.5", "A" vs "(A)", equivalent expressions) but strict on mathematical/factual correctness.
For multiple-choice questions, if the ground truth is an option label such as "A" and the model gives the text/content of that same option instead of the label, mark it correct. If the model gives an option label, use the question/options to verify that the selected option is correct.
Return your judgment as a JSON object with EXACTLY this format (no other text):

{{"verdict": "correct" or "incorrect", "reasoning": "<brief explanation>"}}
"""

PASSK_JUDGE_PROMPT_TEMPLATE = """\
You are an expert answer evaluator. Determine whether the model solved the problem under pass@k evaluation.

## Question and Options:
{question_context}

## Model's Extracted Answers:
{attempt_answers}

## Ground Truth Answer:
{ground_truth}

## Instructions:
Compare each extracted model answer to the ground truth in the context of the question and options.
Return one correctness verdict for every listed attempt, preserving the exact attempt order.
If at least one model answer is correct, set the overall "verdict" to "correct"; otherwise set it to "incorrect".
Be flexible with formatting differences (e.g. "1/2" vs "0.5", "A" vs "(A)", equivalent expressions) but strict on mathematical/factual correctness.
For multiple-choice questions, if the ground truth is an option label such as "A" and a model answer gives the text/content of that same option instead of the label, mark that attempt correct. If a model answer gives an option label, use the question/options to verify that the selected option is correct.
The "attempt_verdicts" array must contain exactly {attempt_count} entries, one per listed attempt.
Return your judgment as a JSON object with EXACTLY this format (no other text):

{{"verdict": "correct" or "incorrect", "attempt_verdicts": ["correct" or "incorrect", ...], "reasoning": "<brief explanation>"}}
"""

VISION_ONLY_PROBLEM_PROMPT = "Please answer the question shown in the image."


def normalize_eval_prompt_mode(value: str | None) -> str:
    mode = str(value or DEFAULT_EVAL_PROMPT_MODE).strip().lower().replace("-", "_")
    aliases = {
        "": DEFAULT_EVAL_PROMPT_MODE,
        "vigos": VIGOS_PROMPT_MODE,
        "vigos_prefill": VIGOS_PROMPT_MODE,
        "student": VIGOS_PROMPT_MODE,
        "student_prefill": VIGOS_PROMPT_MODE,
    }
    try:
        return aliases[mode]
    except KeyError as exc:
        raise ValueError(
            f"Unsupported eval prompt mode {value!r}; expected 'vigos'."
        ) from exc


def prompt_mode_description(prompt_mode: str | None) -> str:
    normalize_eval_prompt_mode(prompt_mode)
    return "ViGOS student prompt plus assistant-side '<description>' prefill"


def assistant_prefill_for_prompt_mode(prompt_mode: str | None) -> str:
    normalize_eval_prompt_mode(prompt_mode)
    return DESCRIPTION_PREFILL


@dataclass(frozen=True)
class DatasetSpec:
    path: str
    split: str

    @property
    def safe_name(self) -> str:
        return sanitize_dataset_name(self.path)


@dataclass(frozen=True)
class EvalSample:
    dataset: str
    sample_id: str
    problem: str
    ground_truth: str
    images: list[Image.Image]
    image_metadata: list[str]
    raw: dict[str, Any]


def parse_dataset_specs(raw: str | None, default_split: str = "test") -> list[DatasetSpec]:
    value = raw or ",".join(DEFAULT_ZLI_DATASETS)
    if value.strip().lower() in {"none", "null", "no", "false", "-"}:
        return []
    specs = []
    for part in value.split(","):
        item = part.strip()
        if not item:
            continue
        if "@" in item:
            path, split = item.rsplit("@", 1)
            split = split.strip() or default_split
        else:
            path, split = item, default_split
        specs.append(DatasetSpec(path=path.strip(), split=split))
    if not specs:
        raise ValueError("No evaluation datasets were configured.")
    return specs


def sanitize_dataset_name(value: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    return safe.strip("_") or "dataset"


def sample_from_record(dataset: str, index: int, record: dict[str, Any]) -> EvalSample:
    images = extract_images(record)
    problem = extract_problem(record)
    ground_truth = extract_ground_truth(record)
    return EvalSample(
        dataset=dataset,
        sample_id=str(record.get("id", record.get("problem_id", index))),
        problem=problem,
        ground_truth=ground_truth,
        images=images,
        image_metadata=extract_image_metadata(record),
        raw=record,
    )


def extract_problem(record: dict[str, Any]) -> str:
    for key in ("problem", "question", "query", "prompt", "text", "instruction"):
        value = record.get(key)
        if value is not None and str(value).strip():
            problem = str(value).strip()
            break
    else:
        if iter_image_values(record):
            problem = VISION_ONLY_PROBLEM_PROMPT
        else:
            raise KeyError("Sample does not contain a problem/question field.")

    if "problem" not in record and record.get("options") is not None:
        options = format_options(record["options"])
        if options and options not in problem:
            problem = f"{problem}\n{options}"
    return problem


def extract_ground_truth(record: dict[str, Any]) -> str:
    for key in ("answer", "solution", "ground_truth", "label", "target"):
        if key in record and record[key] is not None:
            answer = normalize_reference_answer(record[key])
            if answer:
                return answer
    return ""


def format_options(value: Any) -> str:
    parsed = value
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return ""
        try:
            parsed = ast.literal_eval(stripped)
        except (SyntaxError, ValueError):
            return stripped

    if isinstance(parsed, dict):
        return "\n".join(f"{key}. {val}" for key, val in parsed.items())
    if isinstance(parsed, (list, tuple)):
        labels = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        lines = []
        for idx, option in enumerate(parsed):
            label = labels[idx] if idx < len(labels) else str(idx + 1)
            lines.append(f"{label}. {option}")
        return "\n".join(lines)
    return str(parsed).strip()


def extract_images(record: dict[str, Any]) -> list[Image.Image]:
    images: list[Image.Image] = []
    seen_ids: set[int] = set()
    for value in iter_image_values(record):
        for image in image_value_to_pil_list(value):
            marker = id(image)
            if marker not in seen_ids:
                images.append(image)
                seen_ids.add(marker)
    return images


def iter_image_values(record: dict[str, Any]) -> list[Any]:
    values = []
    for key in ("images", "image"):
        if key in record and record[key] is not None:
            values.append(record[key])
    numbered = sorted(
        key
        for key in record
        if re.fullmatch(r"image_\d+", key) and record.get(key) is not None
    )
    values.extend(record[key] for key in numbered)
    return values


def image_value_to_pil_list(value: Any) -> list[Image.Image]:
    if value is None:
        return []
    if isinstance(value, Image.Image):
        return [value.convert("RGB")]
    if isinstance(value, (list, tuple)):
        images = []
        for item in value:
            images.extend(image_value_to_pil_list(item))
        return images
    if isinstance(value, dict):
        if value.get("bytes"):
            with Image.open(BytesIO(value["bytes"])) as image:
                return [image.convert("RGB")]
        if value.get("path"):
            return image_value_to_pil_list(value["path"])
        return []
    if isinstance(value, (str, Path)):
        path = Path(value)
        if not path.exists():
            return []
        with Image.open(path) as image:
            return [image.convert("RGB")]
    return []


def extract_image_metadata(record: dict[str, Any]) -> list[str]:
    metadata: list[str] = []
    for value in iter_image_values(record):
        metadata.extend(image_value_to_metadata(value))
    return metadata


def image_value_to_metadata(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, Image.Image):
        filename = getattr(value, "filename", "")
        return [filename or f"PIL.Image({value.size[0]}x{value.size[1]})"]
    if isinstance(value, (list, tuple)):
        items: list[str] = []
        for item in value:
            items.extend(image_value_to_metadata(item))
        return items
    if isinstance(value, dict):
        if value.get("path"):
            return [str(value["path"])]
        if value.get("bytes"):
            return ["<bytes>"]
        return []
    if isinstance(value, (str, Path)):
        return [str(value)]
    return [type(value).__name__]


def build_eval_messages(
    problem: str,
    images: list[Image.Image],
    *,
    prompt_mode: str | None = None,
) -> list[dict[str, Any]]:
    normalize_eval_prompt_mode(prompt_mode)
    messages: list[dict[str, Any]] = []
    content: list[dict[str, Any]] = []
    for image in images:
        content.append({"type": "image", "image": image})
    text = format_vigos_student_prompt(problem)
    content.append({"type": "text", "text": text})
    messages.append({"role": "user", "content": content})
    return messages


def build_eval_prompt(
    processor: Any,
    problem: str,
    images: list[Image.Image],
    *,
    prompt_mode: str | None = None,
) -> str:
    mode = normalize_eval_prompt_mode(prompt_mode)
    messages = build_eval_messages(problem, images, prompt_mode=mode)
    prompt = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    return prompt + assistant_prefill_for_prompt_mode(mode)


def vllm_request(prompt: str, images: list[Image.Image]) -> dict[str, Any]:
    request: dict[str, Any] = {"prompt": prompt}
    if images:
        request["multi_modal_data"] = {"image": images[0] if len(images) == 1 else images}
    return request


def response_from_completion(completion: str, *, prompt_mode: str | None = None) -> str:
    return assistant_prefill_for_prompt_mode(prompt_mode) + completion


def extract_model_answer(response: str) -> str:
    boxed = extract_boxed_content(response)
    if boxed:
        return boxed.strip()
    answer_matches = re.findall(
        r"<answer>\s*(.*?)\s*</answer>",
        str(response or ""),
        flags=re.IGNORECASE | re.DOTALL,
    )
    if not answer_matches:
        return ""
    answer = str(answer_matches[-1] or "").strip()
    answer = re.sub(r"^final\s+answer\s*:\s*", "", answer, flags=re.IGNORECASE)
    return answer.strip()


def build_judge_messages(
    extracted_answer: str,
    ground_truth: str,
    question_context: str = "",
) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": "You are a precise answer evaluation assistant. Always respond with valid JSON only.",
        },
        {
            "role": "user",
            "content": JUDGE_PROMPT_TEMPLATE.format(
                question_context=question_context.strip() or "(question/options not available)",
                extracted_answer=extracted_answer or "(no answer extracted)",
                ground_truth=ground_truth,
            ),
        },
    ]


def build_passk_judge_messages(
    extracted_answers: list[str],
    ground_truth: str,
    question_context: str = "",
) -> list[dict[str, str]]:
    answers = extracted_answers or [""]
    attempt_answers = "\n".join(
        f"{index + 1}. {answer or '(no answer extracted)'}"
        for index, answer in enumerate(answers)
    )
    return [
        {
            "role": "system",
            "content": "You are a precise answer evaluation assistant. Always respond with valid JSON only.",
        },
        {
            "role": "user",
            "content": PASSK_JUDGE_PROMPT_TEMPLATE.format(
                question_context=question_context.strip() or "(question/options not available)",
                attempt_answers=attempt_answers,
                attempt_count=len(answers),
                ground_truth=ground_truth,
            ),
        },
    ]


def parse_judge_output(text: str) -> dict[str, Any]:
    raw = str(text or "").strip()
    parsed: Any | None = None
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        parsed = None

    if parsed is None:
        fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw, re.DOTALL)
        if fenced:
            try:
                parsed = json.loads(fenced.group(1))
            except json.JSONDecodeError:
                parsed = None

    if parsed is None:
        obj = re.search(r"\{[^{}]*\"verdict\"[^{}]*\}", raw, re.DOTALL)
        if obj:
            try:
                parsed = json.loads(obj.group(0))
            except json.JSONDecodeError:
                parsed = None

    if not isinstance(parsed, dict):
        return {
            "verdict": "incorrect",
            "reasoning": f"Failed to parse judge output: {raw[:200]}",
        }

    verdict = str(parsed.get("verdict", "incorrect")).strip().lower()
    if verdict not in {"correct", "incorrect"}:
        verdict = "incorrect"
    reasoning = str(parsed.get("reasoning", "")).strip()
    result: dict[str, Any] = {"verdict": verdict, "reasoning": reasoning}

    raw_attempt_verdicts = None
    for key in ("attempt_verdicts", "attempts", "attempt_correct", "attempt_correctness"):
        if key in parsed:
            raw_attempt_verdicts = parsed[key]
            break
    if isinstance(raw_attempt_verdicts, list):
        attempt_verdicts = []
        for value in raw_attempt_verdicts:
            if isinstance(value, bool):
                attempt_verdicts.append("correct" if value else "incorrect")
                continue
            normalized = str(value).strip().lower()
            if normalized in {"correct", "true", "yes", "1"}:
                attempt_verdicts.append("correct")
            else:
                attempt_verdicts.append("incorrect")
        result["attempt_verdicts"] = attempt_verdicts
    return result
