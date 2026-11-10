"""ONNX Runtime OCR inference without importing PyTorch."""

from __future__ import annotations

import json
from pathlib import Path
from time import perf_counter

import numpy as np
from PIL import Image

from glyphtype.model_data import IMAGE_HEIGHT, prepare_image
from glyphtype.ocr import OcrError, OcrResult


def decode_numpy(logits: np.ndarray, charset: str, length: int) -> str:
    tokens = logits[0, :length].argmax(axis=-1)
    previous = 0
    characters: list[str] = []
    for token in tokens:
        token = int(token)
        if token and token != previous:
            characters.append(charset[token - 1])
        previous = token
    return "".join(characters)


class OnnxOcr:
    def __init__(self, model_path: Path):
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise OcrError("ONNX Runtime is not installed; install the train extra") from exc
        metadata_path = model_path.with_suffix(".json")
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            if metadata["image_height"] != IMAGE_HEIGHT:
                raise OcrError("ONNX model image height is unsupported")
            self.charset = metadata["charset"]
            self.max_width = metadata["max_width"]
            self.session = ort.InferenceSession(str(model_path),
                                                providers=["CPUExecutionProvider"])
        except (OSError, KeyError, ValueError, RuntimeError) as exc:
            raise OcrError(f"Cannot load ONNX model or metadata: {exc}") from exc

    def recognize(self, image: Image.Image) -> OcrResult:
        array = prepare_image(image, self.max_width)[None, :, :, :]
        start = perf_counter()
        try:
            logits = self.session.run(["logits"], {"images": array})[0]
        except Exception as exc:
            raise OcrError(f"ONNX inference failed: {exc}") from exc
        latency_ms = (perf_counter() - start) * 1000
        text = decode_numpy(logits, self.charset, array.shape[-1] // 4)
        return OcrResult(text=text, confidence=None, latency_ms=latency_ms)
