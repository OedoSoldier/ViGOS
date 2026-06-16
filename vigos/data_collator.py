"""Data collation for ViGOS Qwen VL training."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from PIL import Image

from .answer_utils import normalize_reference_answer


DESCRIPTION_PREFILL = "<description>"
THINK_PREFILL = "<think>"
TRANSITION_PROMPT = (
    "\n\nAfter reading the reference solution above, make sure you truly understand "
    "the reasoning behind each step; do not copy or paraphrase it. Now, using your "
    "own words and independent reasoning, derive the same final answer to the problem above. "
    "Think step by step, explore different approaches, and don't be afraid to backtrack "
    "or reconsider if something doesn't work out:\n"
)


def format_vigos_student_prompt(problem: Any) -> str:
    problem_text = str(problem).strip()
    return (
        f"Problem: {problem_text}\n\n"
        "You are tasked with analyzing an image to generate a detailed description to help you answer the question. "
        "First analyze the image and produce a self-contained description, detailed enough that can lead to the correct answer. Do not include the final answer in the description. Wrap the entire description in <description> </description> tags.\n"
        "Next, engage in an internal dialogue and include self-reflection or verification in your reasoning process. Provide your detailed, step-by-step reasoning based on the image description information and image, and enclose this part within <think> </think> tags.\n"
        "Finally, provide a single word or phrase answer to the question in \\boxed{}.\n"
        "The output format should be: <description> image description here </description> <think> reasoning process here </think> \\boxed{FINAL ANSWER here}."
    )


def format_vigos_reasoning_teacher_prompt(problem: Any, reference_answer: Any) -> str:
    problem_text = str(problem).strip()
    answer_text = str(reference_answer).strip()
    return (
        f"Problem: {problem_text}\n\n"
        f"Here is a reference solution to this problem:\n"
        f"=== Reference Solution Begin ===\n{answer_text}\n=== Reference Solution End ===\n"
        f"{TRANSITION_PROMPT}\n"
        f"Please reason step by step, and put your final answer within \\boxed{{}}."
        "The output format should be: <think> reasoning process here </think> \\boxed{}"
    )


def format_vigos_reference_teacher_prompt(problem: Any, reference_answer: Any) -> str:
    problem_text = str(problem).strip()
    answer_text = str(reference_answer).strip()
    return (
        f"Problem: {problem_text}\n\n"
        f"Here is a reference solution to this problem:\n"
        f"=== Reference Solution Begin ===\n{answer_text}\n=== Reference Solution End ===\n"
        f"{TRANSITION_PROMPT}\n"
        f"Please first describe the visual evidence that can lead to the correct answer. Wrap the entire description in <description> </description> tags.\n"
        f"Then reason step by step, and enclose this part within <think> </think> tags.\nFinally, put your final answer within \\boxed{{}}."
        "The output format should be: <description> image description here </description> <think> reasoning process here </think> \\boxed{FINAL ANSWER here}."
    )


def build_vigos_student_message(problem: Any, image: Image.Image | None) -> list[dict[str, Any]]:
    return _message_with_optional_image(format_vigos_student_prompt(problem), image)


def build_vigos_reasoning_teacher_message(
    problem: Any,
    reference_answer: Any,
    image: Image.Image | None,
) -> list[dict[str, Any]]:
    return _message_with_optional_image(
        format_vigos_reasoning_teacher_prompt(problem, reference_answer),
        image,
    )


def build_vigos_reference_teacher_message(
    problem: Any,
    reference_answer: Any,
    image: Image.Image | None,
) -> list[dict[str, Any]]:
    return _message_with_optional_image(
        format_vigos_reference_teacher_prompt(problem, reference_answer),
        image,
    )


def _as_rgb_image(value: Any) -> Image.Image:
    if isinstance(value, list):
        if not value:
            raise ValueError("Expected at least one image, got an empty list.")
        value = value[0]
    if not isinstance(value, Image.Image):
        raise TypeError(f"Expected a PIL image, got {type(value)!r}")
    return value.convert("RGB")


def _format_reasoning_reference(feature: dict[str, Any], answer_field: str) -> str:
    answer = normalize_reference_answer(feature.get(answer_field))
    if answer:
        return answer
    return "No reference answer was provided."


def _message_with_optional_image(
    text: str, image: Image.Image | None
) -> list[dict[str, Any]]:
    content: list[dict[str, Any]] = []
    if image is not None:
        content.append({"type": "image", "image": image})
    if text:
        content.append({"type": "text", "text": text})
    return [{"role": "user", "content": content}]


@dataclass
class ViGOSDataCollator:
    processor: Any
    max_prompt_length: int = 32768
    answer_field: str = "answer"

    def __post_init__(self) -> None:
        tokenizer = getattr(self.processor, "tokenizer", self.processor)
        if getattr(tokenizer, "pad_token", None) is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, Any]:
        student_messages = []
        perception_messages = []
        reasoning_messages = []
        reference_messages = []
        student_prompt_texts = []
        student_images = []
        problems = []
        references = []
        answers = []
        sample_ids = []

        for local_idx, feature in enumerate(features):
            image = _as_rgb_image(feature.get("images", feature.get("image")))
            problem = str(feature["problem"]).strip()
            reference = _format_reasoning_reference(feature, self.answer_field)
            answer = normalize_reference_answer(feature.get(self.answer_field))
            sample_id = int(feature.get("problem_id", local_idx))

            student_message = build_vigos_student_message(problem, image)
            perception_message = _message_with_optional_image("", image)
            reasoning_message = build_vigos_reasoning_teacher_message(
                problem,
                reference,
                image,
            )
            reference_message = build_vigos_reference_teacher_message(
                problem,
                reference,
                image,
            )

            student_messages.append(student_message)
            perception_messages.append(perception_message)
            reasoning_messages.append(reasoning_message)
            reference_messages.append(reference_message)
            student_prompt_texts.append(
                self.processor.apply_chat_template(
                    student_message,
                    tokenize=False,
                    add_generation_prompt=True,
                )
                + DESCRIPTION_PREFILL
            )
            student_images.append(image)
            problems.append(problem)
            references.append(reference)
            answers.append(answer)
            sample_ids.append(sample_id)

        result: dict[str, Any] = {}
        result.update(
            self._encode(
                "student",
                student_messages,
                assistant_prefill=DESCRIPTION_PREFILL,
            )
        )
        result.update(self._encode("perception", perception_messages))
        result.update(
            self._encode(
                "reasoning",
                reasoning_messages,
                assistant_prefill=THINK_PREFILL,
            )
        )
        result.update(
            self._encode(
                "reference",
                reference_messages,
                assistant_prefill=DESCRIPTION_PREFILL,
            )
        )
        result["student_prompt_texts"] = student_prompt_texts
        result["student_images"] = student_images
        result["vigos_problems"] = problems
        result["vigos_references"] = references
        result["vigos_answers"] = answers
        result["sample_ids"] = torch.tensor(sample_ids, dtype=torch.long)
        return result

    def _encode(
        self,
        prefix: str,
        messages: list[list[dict[str, Any]]],
        *,
        assistant_prefill: str | None = None,
    ) -> dict[str, torch.Tensor]:
        tokenizer = getattr(self.processor, "tokenizer", self.processor)
        prefill_ids = _plain_token_ids(tokenizer, assistant_prefill)
        max_length = self.max_prompt_length
        if prefill_ids:
            max_length -= len(prefill_ids)
            if max_length <= 0:
                raise ValueError(
                    "max_prompt_length is too small for the requested assistant prefill."
                )
        encoded = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        )
        if prefill_ids:
            encoded = _append_token_suffix(encoded, prefill_ids)
        return {f"{prefix}_prompt_{key}": value for key, value in encoded.items()}


def _plain_token_ids(tokenizer: Any, text: str | None) -> list[int]:
    if not text:
        return []
    if hasattr(tokenizer, "encode"):
        try:
            return list(tokenizer.encode(text, add_special_tokens=False))
        except TypeError:
            return list(tokenizer.encode(text))
    encoded = tokenizer(text, add_special_tokens=False)
    ids = encoded["input_ids"] if isinstance(encoded, dict) else encoded.input_ids
    if ids and isinstance(ids[0], list):
        ids = ids[0]
    return list(ids)


def _append_token_suffix(
    encoded: dict[str, torch.Tensor],
    suffix_ids: list[int],
) -> dict[str, torch.Tensor]:
    suffix = torch.tensor(
        suffix_ids,
        dtype=encoded["input_ids"].dtype,
        device=encoded["input_ids"].device,
    ).unsqueeze(0)
    suffix = suffix.expand(encoded["input_ids"].shape[0], -1)
    encoded = dict(encoded)
    encoded["input_ids"] = torch.cat([encoded["input_ids"], suffix], dim=1)
    suffix_attention = torch.ones(
        suffix.shape,
        dtype=encoded["attention_mask"].dtype,
        device=encoded["attention_mask"].device,
    )
    encoded["attention_mask"] = torch.cat(
        [encoded["attention_mask"], suffix_attention],
        dim=1,
    )
    return encoded
