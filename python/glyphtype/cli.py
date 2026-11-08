"""CLI for display diagnostics and one-shot region capture."""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
import sys

from glyphtype.capture import CaptureBackend, CaptureError, X11MSSCapture
from glyphtype.config import ConfigError, load_config, load_ocr_config
from glyphtype.display import DisplayError, detect_display, require_x11
from glyphtype.metrics import character_error_rate
from glyphtype.ocr import OcrError, TesseractOcr, recognize_file


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="glyphtype")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("doctor", help="Report display and capture support")
    capture = commands.add_parser("capture", help="Save one configured screen region")
    capture.add_argument("--config", type=Path, default=Path("config.toml"))
    capture.add_argument("--output", type=Path, required=True)
    ocr = commands.add_parser("ocr", help="Recognize text in an existing image")
    ocr.add_argument("--image", type=Path, required=True)
    ocr.add_argument("--config", type=Path, help="Optional TOML OCR settings")
    ocr.add_argument("--expected-file", type=Path, help="Reference text for character accuracy")
    ocr.add_argument("--save-mismatch", type=Path, help="Copy source image here if recognition differs")
    return parser


def main(argv: list[str] | None = None, backend: CaptureBackend | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "doctor":
        display = detect_display()
        print(f"Display session: {display}")
        try:
            require_x11("auto")
        except DisplayError as exc:
            print(f"Capture unavailable: {exc}")
            return 1
        if importlib.util.find_spec("mss") is None:
            print("Capture unavailable: MSS is not installed; install the project package")
            return 1
        print("X11 capture supported (run 'glyphtype capture' to verify access)")
        return 0

    if args.command == "ocr":
        try:
            if args.save_mismatch is not None and args.expected_file is None:
                raise ConfigError("--save-mismatch requires --expected-file")
            settings = load_ocr_config(args.config)
            result = recognize_file(args.image, TesseractOcr(settings))
            if args.expected_file is not None:
                expected = args.expected_file.read_text(encoding="utf-8").rstrip("\n")
                rate = character_error_rate(expected, result.text)
                print(f"CER: {rate:.4f}; character accuracy: {max(0, 1-rate):.2%}", file=sys.stderr)
                if rate and args.save_mismatch is not None:
                    with args.save_mismatch.open("xb") as target:
                        target.write(args.image.read_bytes())
                    print(f"Saved mismatch image: {args.save_mismatch}", file=sys.stderr)
            print(f"OCR latency: {result.latency_ms:.1f} ms", file=sys.stderr)
            print(result.text)
            return 0
        except (ConfigError, OcrError, OSError, UnicodeError) as exc:
            print(f"Error: {exc}", file=sys.stderr)
            return 1

    try:
        config = load_config(args.config)
        require_x11(config.backend)
        if args.output.exists():
            raise CaptureError(f"Output already exists: {args.output}")
        png = (backend or X11MSSCapture()).capture_png(config.region)
        with args.output.open("xb") as stream:
            stream.write(png)
    except (ConfigError, DisplayError, CaptureError, OSError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    print(f"Saved {args.output} ({config.region.width}x{config.region.height})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
