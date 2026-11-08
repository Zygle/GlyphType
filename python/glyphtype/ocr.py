"""Interchangeable OCR contract and local Tesseract implementation."""

from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
import subprocess
from time import perf_counter
from typing import Protocol

from PIL import Image, UnidentifiedImageError

from glyphtype.config import OcrConfig
from glyphtype.preprocessing import preprocess


class OcrError(RuntimeError):
    """An image cannot be read or recognized."""


@dataclass(frozen=True)
class OcrResult:
    text: str
    confidence: float | None
    latency_ms: float


class OcrEngine(Protocol):
    def recognize(self, image: Image.Image) -> OcrResult:
        """Recognize text from an image."""


class TesseractOcr:
    def __init__(self, settings: OcrConfig):
        self.settings = settings

    def recognize(self, image: Image.Image) -> OcrResult:
        prepared = preprocess(image, self.settings.preprocessing)
        buffer = BytesIO()
        prepared.save(buffer, format="PNG")
        command = [self.settings.executable, "stdin", "stdout", "-l",
                   self.settings.language, "--psm", str(self.settings.page_segmentation_mode)]
        start = perf_counter()
        try:
            completed = subprocess.run(
                command, input=buffer.getvalue(), stdout=subprocess.PIPE,
                stderr=subprocess.PIPE, timeout=30, check=False,
            )
        except FileNotFoundError as exc:
            raise OcrError(f"Tesseract executable not found: {self.settings.executable}") from exc
        except subprocess.TimeoutExpired as exc:
            raise OcrError("Tesseract timed out after 30 seconds") from exc
        except OSError as exc:
            raise OcrError(f"Could not start Tesseract: {exc}") from exc
        latency_ms = (perf_counter() - start) * 1000
        if completed.returncode:
            detail = completed.stderr.decode("utf-8", errors="replace").strip()
            raise OcrError(f"Tesseract failed: {detail or f'exit code {completed.returncode}'}")
        text = completed.stdout.decode("utf-8", errors="replace").rstrip("\n\f")
        return OcrResult(text=text, confidence=None, latency_ms=latency_ms)


def recognize_file(path: Path, engine: OcrEngine) -> OcrResult:
    try:
        with Image.open(path) as image:
            image.load()
            return engine.recognize(image)
    except (UnidentifiedImageError, OSError) as exc:
        raise OcrError(f"Cannot read image {path}: {exc}") from exc
