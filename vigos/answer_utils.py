"""Answer normalization helpers for reference fields."""

from __future__ import annotations

import re
from typing import Any

try:
    from mathruler.grader import (
        extract_boxed_content as mathruler_extract_boxed_content,
        grade_answer as mathruler_grade_answer,
    )
except ImportError:  # pragma: no cover - release users may inspect help before installing extras.
    mathruler_extract_boxed_content = None
    mathruler_grade_answer = None


def extract_boxed_content(value: Any) -> str | None:
    """Return the verifier's boxed-answer payload, if present."""

    text_value = str(value or "")
    if mathruler_extract_boxed_content is not None:
        content = mathruler_extract_boxed_content(text_value)
    else:
        content = _fallback_extract_boxed_content(text_value)
    if content is None:
        return None
    text = str(content).strip()
    if not text or text.lower() == "none":
        return None
    return text


def grade_answer(prediction: Any, reference: Any) -> bool:
    """Grade answers with the same verifier used during training."""

    if prediction is None or reference is None:
        return False
    if mathruler_grade_answer is not None:
        return bool(mathruler_grade_answer(str(prediction), str(reference)))
    return str(prediction).strip().casefold() == str(reference).strip().casefold()


def normalize_reference_answer(value: Any) -> str:
    """Normalize dataset answers before placing them inside a teacher prompt."""
    text = str(value or "").strip()
    text = re.sub(r"^```(?:\w+)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    text = re.sub(
        r"^answer\s*[:：]\s*",
        "",
        text.strip(),
        flags=re.I,
    )

    answer_match = re.search(
        r"<answer>\s*(.*?)\s*</answer>",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if answer_match:
        text = answer_match.group(1).strip()

    boxed = extract_boxed_content(text)
    if boxed is not None:
        text = boxed

    return text.strip().strip("`'\"")


def _fallback_extract_boxed_content(text: str) -> str | None:
    marker = "\\boxed{"
    start = text.rfind(marker)
    if start < 0:
        return None
    cursor = start + len(marker)
    depth = 1
    content_start = cursor
    while cursor < len(text):
        char = text[cursor]
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[content_start:cursor]
        cursor += 1
    return None
