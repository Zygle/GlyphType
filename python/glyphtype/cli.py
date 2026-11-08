"""CLI for display diagnostics and one-shot region capture."""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
import sys

from glyphtype.capture import CaptureBackend, CaptureError, X11MSSCapture
from glyphtype.config import ConfigError, load_config
from glyphtype.display import DisplayError, detect_display, require_x11


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="glyphtype")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("doctor", help="Report display and capture support")
    capture = commands.add_parser("capture", help="Save one configured screen region")
    capture.add_argument("--config", type=Path, default=Path("config.toml"))
    capture.add_argument("--output", type=Path, required=True)
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
