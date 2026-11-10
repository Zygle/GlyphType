"""Interchangeable OCR contract and local Tesseract implementation."""

from __future__ import annotations

from dataclasses import dataclass
import csv
from io import BytesIO
import os
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
    def __init__(self, settings: OcrConfig, include_confidence: bool = False):
        self.settings = settings
        self.include_confidence = include_confidence

    def _tessdata_dir(self) -> Path | None:
        languages = self.settings.language.split("+")
        if self.settings.tessdata_dir:
            directory = Path(self.settings.tessdata_dir)
            missing = [name for name in languages
                       if not (directory / f"{name}.traineddata").is_file()]
            if missing:
                raise OcrError(
                    f"Tesseract language data missing in {directory}: {', '.join(missing)}"
                )
            return directory
        if "TESSDATA_PREFIX" in os.environ:
            return None  # Respect the caller's explicit Tesseract setting.
        for directory in (Path.cwd() / "models" / "tessdata",
                          Path(__file__).resolve().parents[2] / "models" / "tessdata"):
            if all((directory / f"{name}.traineddata").is_file() for name in languages):
                return directory
        return None

    def _invoke(self, image_bytes: bytes, tsv: bool = False) -> bytes:
        command = [self.settings.executable, "stdin", "stdout", "-l",
                   self.settings.language, "--psm", str(self.settings.page_segmentation_mode)]
        tessdata_dir = self._tessdata_dir()
        if tessdata_dir is not None:
            command.extend(("--tessdata-dir", str(tessdata_dir)))
        if tsv:
            command.append("tsv")
        try:
            completed = subprocess.run(
                command, input=image_bytes, stdout=subprocess.PIPE,
                stderr=subprocess.PIPE, timeout=30, check=False,
            )
        except FileNotFoundError as exc:
            raise OcrError(f"Tesseract executable not found: {self.settings.executable}") from exc
        except subprocess.TimeoutExpired as exc:
            raise OcrError("Tesseract timed out after 30 seconds") from exc
        except OSError as exc:
            raise OcrError(f"Could not start Tesseract: {exc}") from exc
        if completed.returncode:
            detail = completed.stderr.decode("utf-8", errors="replace").strip()
            if "Error opening data file" in detail or "Failed loading language" in detail:
                raise OcrError(
                    f"Tesseract cannot load {self.settings.language!r} language data. "
                    "Install that language's traineddata or set [ocr] tessdata_dir "
                    f"to its directory. Details: {detail}"
                )
            raise OcrError(f"Tesseract failed: {detail or f'exit code {completed.returncode}'}")
        return completed.stdout

    def recognize(self, image: Image.Image) -> OcrResult:
        prepared = preprocess(image, self.settings.preprocessing)
        buffer = BytesIO()
        prepared.save(buffer, format="PNG")
        start = perf_counter()
        image_bytes = buffer.getvalue()
        text = self._invoke(image_bytes).decode("utf-8", errors="replace").rstrip("\n\f")
        confidence = None
        if self.include_confidence:
            tsv = self._invoke(image_bytes, tsv=True).decode("utf-8", errors="replace")
            try:
                scores = [float(row["conf"]) for row in csv.DictReader(tsv.splitlines(), delimiter="\t")
                          if row.get("text", "").strip() and float(row["conf"]) >= 0]
            except (KeyError, ValueError, TypeError) as exc:
                raise OcrError("Could not parse Tesseract confidence output") from exc
            confidence = min(scores) if scores else None
        latency_ms = (perf_counter() - start) * 1000
        return OcrResult(text=text, confidence=confidence, latency_ms=latency_ms)


def recognize_file(path: Path, engine: OcrEngine) -> OcrResult:
    try:
        with Image.open(path) as image:
            image.load()
            return engine.recognize(image)
    except (UnidentifiedImageError, OSError) as exc:
        raise OcrError(f"Cannot read image {path}: {exc}") from exc
