"""Export a trusted PyTorch checkpoint to verified ONNX for CPU inference."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch

from glyphtype.model import LineRecognizer
from glyphtype.model_data import IMAGE_HEIGHT


def export(checkpoint_path: Path, output: Path) -> None:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model = LineRecognizer(len(checkpoint["charset"]) + 1,
                           checkpoint["hidden_size"])
    model.load_state_dict(checkpoint["model"])
    model.eval()
    output.parent.mkdir(parents=True, exist_ok=True)
    example = torch.zeros(1, 1, IMAGE_HEIGHT, 256)
    torch.onnx.export(
        model, example, str(output), opset_version=17, dynamo=False,
        input_names=["images"], output_names=["logits"],
        dynamic_axes={"images": {3: "width"}, "logits": {1: "time"}},
    )
    onnx.checker.check_model(onnx.load(str(output)))
    session = ort.InferenceSession(str(output), providers=["CPUExecutionProvider"])
    for width in (256, 320):
        sample = torch.rand(1, 1, IMAGE_HEIGHT, width)
        with torch.inference_mode():
            expected = model(sample).numpy()
        actual = session.run(["logits"], {"images": sample.numpy()})[0]
        np.testing.assert_allclose(actual, expected, rtol=1e-3, atol=1e-3)
    metadata = {
        "charset": checkpoint["charset"], "image_height": IMAGE_HEIGHT,
        "max_width": checkpoint["max_width"], "hidden_size": checkpoint["hidden_size"],
        "checkpoint_epoch": checkpoint["epoch"],
    }
    output.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n",
                                           encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=Path("models/line_ocr/best.pt"))
    parser.add_argument("--output", type=Path, default=Path("models/line_ocr/model.onnx"))
    args = parser.parse_args(argv)
    try:
        export(args.checkpoint, args.output)
    except (OSError, ValueError, RuntimeError, KeyError, AssertionError) as exc:
        parser.exit(1, f"Error: {exc}\n")
    print(f"Verified ONNX model: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
