"""Compare ONNX line OCR and Tesseract on the same labeled test images."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import resource
import statistics

from glyphtype.config import OcrConfig
from glyphtype.dataset import LineDataset
from glyphtype.metrics import character_error_rate, word_error_rate
from glyphtype.ocr import OcrError, OcrResult, TesseractOcr
from glyphtype.onnx_ocr import OnnxOcr


def summarize(references: list[str], predictions: list[str], latencies: list[float]) -> dict:
    chars = sum(len(text) for text in references)
    words = sum(len(text.split()) for text in references)
    character_errors = sum(character_error_rate(ref, pred) * len(ref)
                           for ref, pred in zip(references, predictions))
    word_errors = sum(word_error_rate(ref, pred) * len(ref.split())
                      for ref, pred in zip(references, predictions))
    ordered = sorted(latencies)
    return {
        "lines": len(references),
        "cer": character_errors / chars if chars else None,
        "wer": word_errors / words if words else None,
        "exact_line_accuracy": sum(ref == pred for ref, pred in zip(references, predictions)) / len(references),
        "mean_latency_ms": statistics.mean(latencies),
        "p95_latency_ms": ordered[min(len(ordered) - 1, int(len(ordered) * 0.95))],
    }


def evaluate(dataset_root: Path, model_path: Path, language: str,
             limit: int | None = None) -> dict:
    dataset = LineDataset(dataset_root, "test")
    if not dataset:
        raise ValueError("Test split is empty")
    count = min(len(dataset), limit) if limit is not None else len(dataset)
    if count < 1:
        raise ValueError("Limit must be positive")
    engines = {
        "onnx": OnnxOcr(model_path),
        "tesseract": TesseractOcr(OcrConfig(language=language)),
    }
    references: list[str] = []
    predictions: dict[str, list[str]] = {name: [] for name in engines}
    latencies: dict[str, list[float]] = {name: [] for name in engines}
    for index in range(count):
        image, label = dataset[index]
        references.append(label)
        try:
            for name, engine in engines.items():
                result: OcrResult = engine.recognize(image)
                predictions[name].append(result.text)
                latencies[name].append(result.latency_ms)
        finally:
            image.close()
    results = {name: summarize(references, predictions[name], latencies[name])
               for name in engines}
    results["recommendation"] = (
        "onnx" if results["onnx"]["cer"] < results["tesseract"]["cer"]
        and results["onnx"]["exact_line_accuracy"] >= results["tesseract"]["exact_line_accuracy"]
        else "tesseract"
    )
    results["peak_process_rss_kib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    results["peak_tesseract_child_rss_kib"] = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
    results["dataset_root"] = str(dataset_root)
    results["model_path"] = str(model_path)
    return results


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("datasets/synthetic"))
    parser.add_argument("--model", type=Path, default=Path("models/line_ocr/model.onnx"))
    parser.add_argument("--language", default="eng")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--output", type=Path, default=Path("models/line_ocr/evaluation.json"))
    args = parser.parse_args(argv)
    try:
        results = evaluate(args.dataset, args.model, args.language, args.limit)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    except (OSError, ValueError, OcrError, RuntimeError) as exc:
        parser.exit(1, f"Error: {exc}\n")
    print(json.dumps(results, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
