"""Dataset loading and filtering helpers for ViGOS."""

from __future__ import annotations

from io import BytesIO
from typing import Any

from datasets import Dataset, load_dataset
from PIL import Image


def load_vigos_dataset(dataset_name: str, split: str = "train") -> Dataset:
    """Load a ViGOS training dataset directly from HuggingFace Datasets."""
    if not dataset_name or not dataset_name.strip():
        raise ValueError(
            "dataset_name must be a HuggingFace dataset id, for example "
            "'your-org/your-vigos-dataset'."
        )
    return load_dataset(dataset_name, split=split)


def filter_tiny_image_samples(dataset: Dataset, min_image_size: int) -> Dataset:
    if min_image_size < 1:
        raise ValueError(f"min_image_size must be positive, got {min_image_size}.")
    image_column = image_column_name(dataset)
    return dataset.filter(
        lambda image: has_valid_image_size(image, min_image_size=min_image_size),
        input_columns=[image_column],
        desc=f"Filtering images smaller than {min_image_size}px",
    )


def image_column_name(dataset: Dataset) -> str:
    for column in ("images", "image"):
        if column in dataset.column_names:
            return column
    raise ValueError(
        "ViGOS training requires an image column named either 'images' or 'image'."
    )


def has_valid_image_size(value: Any, *, min_image_size: int) -> bool:
    size = image_size(value)
    if size is None:
        return False
    width, height = size
    return width >= min_image_size and height >= min_image_size


def image_size(value: Any) -> tuple[int, int] | None:
    if isinstance(value, list):
        if not value:
            return None
        value = value[0]
    if isinstance(value, Image.Image):
        return value.size
    if isinstance(value, dict):
        image_bytes = value.get("bytes")
        if image_bytes is not None:
            try:
                with Image.open(BytesIO(image_bytes)) as image:
                    return image.size
            except Exception:
                return None
        image_path = value.get("path")
        if image_path:
            try:
                with Image.open(image_path) as image:
                    return image.size
            except Exception:
                return None
    return None
