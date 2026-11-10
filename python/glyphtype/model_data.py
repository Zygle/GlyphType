"""Shared image and label encoding for training and ONNX inference."""

from __future__ import annotations

import numpy as np
from PIL import Image
import string

IMAGE_HEIGHT = 32
MAX_WIDTH = 1024
DEFAULT_CHARSET = string.digits + string.ascii_uppercase + string.ascii_lowercase + " " + string.punctuation


def prepare_image(image: Image.Image, max_width: int = MAX_WIDTH) -> np.ndarray:
    """Return inverted grayscale float32 pixels shaped 1,H,W."""
    gray = image.convert("L")
    width = max(4, round(gray.width * IMAGE_HEIGHT / gray.height))
    if width > max_width:
        width = max_width
    resized = gray.resize((width, IMAGE_HEIGHT), Image.Resampling.BILINEAR)
    return ((255.0 - np.asarray(resized, dtype=np.float32)) / 255.0)[None, :, :]


def charset_from_labels(labels: list[str]) -> str:
    characters = sorted(set("".join(labels)))
    if not characters or any(char in "\r\n" for char in characters):
        raise ValueError("Training labels must be nonempty single lines")
    return "".join(characters)


def collate_samples(samples: list[tuple[Image.Image, str]], charset: str,
                    max_width: int = MAX_WIDTH):
    """Pad widths only within a batch; keep labels as CTC targets."""
    import torch

    arrays = [prepare_image(image, max_width) for image, _ in samples]
    widths = [array.shape[-1] for array in arrays]
    batch = np.zeros((len(arrays), 1, IMAGE_HEIGHT, max(widths)), dtype=np.float32)
    for index, array in enumerate(arrays):
        batch[index, :, :, :array.shape[-1]] = array
    lookup = {char: index + 1 for index, char in enumerate(charset)}
    labels = [text for _, text in samples]
    try:
        encoded = [[lookup[char] for char in text] for text in labels]
    except KeyError as exc:
        raise ValueError(f"Label contains character absent from charset: {exc}") from exc
    for text, width in zip(labels, widths):
        repeated = sum(a == b for a, b in zip(text, text[1:]))
        if width // 4 < len(text) + repeated:
            raise ValueError(f"Image too narrow for CTC label {text!r}; raise max_width")
    return (torch.from_numpy(batch), torch.tensor(widths, dtype=torch.long),
            torch.tensor([item for line in encoded for item in line], dtype=torch.long),
            torch.tensor([len(line) for line in encoded], dtype=torch.long), labels)
