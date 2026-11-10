"""Lazy reader for labeled synthetic OCR images."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

from PIL import Image


@dataclass(frozen=True)
class DatasetRecord:
    image: str
    text: str
    kind: str
    font: str


class LineDataset:
    """Keep only small manifest records in memory; load image pixels on demand."""

    def __init__(self, root: Path, split: str = "train") -> None:
        if split not in ("train", "validation", "test"):
            raise ValueError("split must be train, validation, or test")
        self.root = root
        manifest = root / f"{split}.jsonl"
        with manifest.open(encoding="utf-8") as stream:
            self.records = []
            for line in stream:
                if line.strip():
                    row = json.loads(line)
                    self.records.append(DatasetRecord(
                        image=row["image"], text=row["text"],
                        kind=row["kind"], font=row["font"],
                    ))

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> tuple[Image.Image, str]:
        record = self.records[index]
        with Image.open(self.root / record.image) as image:
            return image.convert("L"), record.text
