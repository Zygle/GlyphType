"""CLI for display diagnostics and one-shot region capture."""

from __future__ import annotations

import argparse
from contextlib import nullcontext
from dataclasses import replace
from io import BytesIO
import importlib.util
from pathlib import Path
import shutil
import sys
from threading import Event

from PIL import Image

from glyphtype.capture import CaptureBackend, CaptureError, KWinWaylandCapture, X11MSSCapture
from glyphtype.config import ConfigError, OcrConfig, Region, load_config, load_ocr_config, load_tracking_config, load_typing_config
from glyphtype.display import DisplayError, detect_display, resolve_backend
from glyphtype.metrics import character_error_rate
from glyphtype.ocr import OcrEngine, OcrError, TesseractOcr, recognize_file
from glyphtype.selection import SelectionError, capture_wayland_selection, select_x11_region
from glyphtype.tracking import State, TextTracker, TrackingError
from glyphtype.typing import EscapeStop, TypingController, TypingError, XdotoolBackend, normalize_typing_text
from glyphtype.wayland_typing import KdeWaylandInput


def capture_backend_for(display: str) -> CaptureBackend:
    return KWinWaylandCapture() if display == "wayland" else X11MSSCapture()


def capture_source(config_path: Path, select: bool,
                   backend: CaptureBackend | None = None) -> tuple[bytes, tuple[int, int]]:
    """Capture one native crop, optionally selected by the user."""
    config = load_config(config_path)
    display = resolve_backend(config.backend)
    if select and display == "wayland":
        png = capture_wayland_selection()
        with Image.open(BytesIO(png)) as image:
            return png, image.size
    region = select_x11_region() if select else config.region
    png = (backend or capture_backend_for(display)).capture_png(region)
    with Image.open(BytesIO(png)) as image:
        return png, image.size


def make_ocr_engine(settings: OcrConfig) -> OcrEngine:
    if settings.backend == "onnx":
        from glyphtype.onnx_ocr import OnnxOcr
        return OnnxOcr(Path(settings.model_path))
    return TesseractOcr(settings)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="glyphtype")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("doctor", help="Report display and capture support")
    capture = commands.add_parser("capture", help="Save one configured screen region")
    capture.add_argument("--config", type=Path, default=Path("config.toml"))
    capture.add_argument("--output", type=Path, required=True)
    capture.add_argument("--select", action="store_true", help="Drag a region using the native desktop selector")
    ocr = commands.add_parser("ocr", help="Recognize text in an existing image")
    ocr.add_argument("--image", type=Path, required=True)
    ocr.add_argument("--config", type=Path, help="Optional TOML OCR settings")
    ocr.add_argument("--expected-file", type=Path, help="Reference text for character accuracy")
    ocr.add_argument("--save-mismatch", type=Path, help="Copy source image here if recognition differs")
    typing = commands.add_parser("type", help="OCR one image or capture, then optionally type it")
    source = typing.add_mutually_exclusive_group(required=True)
    source.add_argument("--image", type=Path, help="Existing text image")
    source.add_argument("--capture", action="store_true", help="Capture configured rectangle")
    source.add_argument("--select", action="store_true", help="Drag and capture a rectangle")
    typing.add_argument("--config", type=Path, default=Path("config.toml"))
    typing.add_argument("--start", action="store_true", help="Explicitly start this typing run")
    typing.add_argument("--live", action="store_true", help="Enable real keyboard input on a supported backend")
    typing.add_argument("--target-window", type=lambda value: int(value, 0),
                        help="Selected X11 window ID from xdotool getactivewindow")
    typing.add_argument("--focus-wait", type=int, default=3,
                        help="Seconds to refocus the target after OCR (0-10; default 3)")
    typing.add_argument("--expected-file", type=Path,
                        help="Optional reference; cancel live typing if OCR differs")
    watch = commands.add_parser("watch", help="Track a fixed text region over repeated captures")
    watch.add_argument("--config", type=Path, default=Path("config.toml"))
    watch.add_argument("--select", action="store_true", help="Drag the region once before watching (X11 only)")
    watch.add_argument("--start", action="store_true", help="Explicitly start watching")
    watch.add_argument("--live", action="store_true", help="Enable real keyboard input on a supported backend")
    watch.add_argument("--target-window", type=lambda value: int(value, 0))
    watch.add_argument("--max-frames", type=int, default=0,
                       help="Stop after this many captures; 0 means until stopped")
    return parser


def run_watch(args: argparse.Namespace) -> int:
    if not args.start:
        raise TypingError("watch requires --start; no input was sent")
    if args.max_frames < 0:
        raise ConfigError("--max-frames must be nonnegative")
    capture_settings = load_config(args.config)
    display = resolve_backend(capture_settings.backend)
    ocr_settings = load_ocr_config(args.config)
    if ocr_settings.backend != "tesseract":
        raise ConfigError("watch requires Tesseract confidence; set ocr.backend = 'tesseract'")
    typing_settings = load_typing_config(args.config)
    tracking_settings = load_tracking_config(args.config)
    stop = Event()
    retry = Event()
    accept = Event()
    if args.live:
        if typing_settings.dry_run:
            raise TypingError("Live watch requires typing.dry_run = false")
        if display == "wayland":
            input_backend = None
            context = KdeWaylandInput(stop, retry, accept)
        else:
            if args.target_window is None:
                raise TypingError("X11 live watch requires --target-window")
            input_backend = XdotoolBackend(args.target_window)
            context = EscapeStop(stop, retry, accept)
    else:
        input_backend = None
        context = nullcontext()
    if args.select and display == "wayland":
        raise SelectionError("Wayland watch needs fixed [capture] coordinates; use capture --select to inspect a crop")
    region = select_x11_region() if args.select else capture_settings.region
    print(f"Watching region {region}; press Ctrl-C to stop", file=sys.stderr)
    tracker = TextTracker(tracking_settings.minimum_confidence, tracking_settings.confirmations)
    engine = TesseractOcr(ocr_settings, include_confidence=True)
    capture_backend = capture_backend_for(display)
    focused_once = False
    frames = 0
    with context as emergency_stop:
        if args.live and display == "wayland":
            input_backend = emergency_stop
            print("Click the authorized target window to select it.", file=sys.stderr)
            title = input_backend.select_target()
            print(f"Selected target: {title}", file=sys.stderr)
        def recover() -> bool:
            nonlocal focused_once
            print(f"Paused: {tracker.reason}. Last OCR text: {result.text!r}", file=sys.stderr)
            if input_backend is None or (args.max_frames and frames >= args.max_frames):
                return False
            retry.clear()
            accept.clear()
            if display == "wayland":
                print("Meta+Shift+F8 retries. Meta+Shift+F9 accepts the reviewed new "
                      "passage. Meta+Shift+F12 stops.", file=sys.stderr)
            else:
                print("F8 retries/resumes. F9 accepts the reviewed new passage. "
                      "Escape stops.", file=sys.stderr)
            while not stop.is_set():
                emergency_stop.verify_running()
                if retry.is_set():
                    retry.clear()
                    tracker.resume()
                    focused_once = False
                    return True
                if accept.is_set():
                    accept.clear()
                    if (tracker.reason or "").startswith("Displayed text changed") and (
                        result.confidence is not None
                        and result.confidence >= tracking_settings.minimum_confidence
                    ):
                        tracker.accept_new_text(result.text)
                        focused_once = False
                        return True
                    print("Cannot accept this frame; press F8 to retry OCR.", file=sys.stderr)
                stop.wait(0.1)
            return False

        while not stop.is_set():
            tracker.begin_capture()
            try:
                png = capture_backend.capture_png(region)
                tracker.begin_recognition()
                with Image.open(BytesIO(png)) as image:
                    result = engine.recognize(image)
                result = replace(result, text=normalize_typing_text(result.text))
            except (CaptureError, OcrError, OSError) as exc:
                tracker.fail(str(exc))
                raise
            tracker.observe(result)
            frames += 1
            print(f"Frame {frames}: {tracker.state.value}, confidence={result.confidence}, "
                  f"OCR={result.latency_ms:.1f} ms", file=sys.stderr)
            if tracker.state == State.PAUSED:
                if recover():
                    continue
                return 130 if stop.is_set() else 2
            if tracker.state == State.READY:
                pending = tracker.pending_text()
                if input_backend is None:
                    print(f"DRY RUN would type: {pending!r}")
                    tracker.commit(len(pending))
                else:
                    sent = [0]
                    try:
                        if not focused_once:
                            shortcut = "Meta+Shift+F12" if display == "wayland" else "Escape"
                            print(f"Refocus the target window within 3 seconds. Press {shortcut} to stop.",
                                  file=sys.stderr)
                            if stop.wait(3):
                                tracker.commit(0)
                                return 130
                            focused_once = True
                        emergency_stop.verify_running()
                        input_backend.verify_target()
                        outcome = TypingController(
                            input_backend, stop, typing_settings.character_delay_ms,
                            safety_check=emergency_stop.verify_running,
                            on_sent=lambda count: sent.__setitem__(0, count),
                        ).type_text(pending)
                    except TypingError as exc:
                        tracker.commit(sent[0])
                        tracker.pause(str(exc))
                        if recover():
                            continue
                        return 130 if stop.is_set() else 2
                    tracker.commit(outcome.typed_count)
                    print(f"Sent {tracker.typed_count}/{len(tracker.text)} characters",
                          file=sys.stderr)
                    if outcome.stopped:
                        return 130
            if args.max_frames and frames >= args.max_frames:
                return 0
            stop.wait(tracking_settings.capture_interval_ms / 1000)
    return 130


def main(argv: list[str] | None = None, backend: CaptureBackend | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "watch":
        try:
            return run_watch(args)
        except KeyboardInterrupt:
            print("Stopped by Ctrl-C", file=sys.stderr)
            return 130
        except (ConfigError, DisplayError, CaptureError, SelectionError, OcrError,
                TypingError, TrackingError, OSError) as exc:
            print(f"Error: {exc}", file=sys.stderr)
            return 1
    if args.command == "doctor":
        display = detect_display()
        print(f"Display session: {display}")
        try:
            resolve_backend("auto")
        except DisplayError as exc:
            print(f"Capture unavailable: {exc}")
            return 1
        dependency = "dbus_next" if display == "wayland" else "mss"
        if importlib.util.find_spec(dependency) is None:
            print(f"Capture unavailable: {dependency} is not installed; install the project package")
            return 1
        if display == "wayland":
            try:
                KWinWaylandCapture().capture_png(Region(0, 0, 1, 1))
            except (CaptureError, ConfigError) as exc:
                print(f"Capture unavailable: {exc}")
                return 1
            print("KDE Wayland exact-region capture available")
            print("Drag selection available" if shutil.which("spectacle") else
                  "Drag selection unavailable: install KDE Spectacle")
        else:
            print("X11 capture available (run 'glyphtype capture' to verify access)")
        return 0

    if args.command == "ocr":
        try:
            if args.save_mismatch is not None and args.expected_file is None:
                raise ConfigError("--save-mismatch requires --expected-file")
            settings = load_ocr_config(args.config)
            result = recognize_file(args.image, make_ocr_engine(settings))
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

    if args.command == "type":
        if not args.start:
            print("Error: typing requires --start; no input was sent", file=sys.stderr)
            return 1
        if not 0 <= args.focus_wait <= 10:
            print("Error: --focus-wait must be from 0 to 10 seconds", file=sys.stderr)
            return 1
        try:
            ocr_settings = load_ocr_config(args.config)
            engine = make_ocr_engine(ocr_settings)
            typing_settings = load_typing_config(args.config)
            if args.live:
                if typing_settings.dry_run:
                    raise TypingError("Live typing requires typing.dry_run = false in config")
                display = resolve_backend(load_config(args.config).backend)
                if display == "x11" and args.target_window is None:
                    raise TypingError("X11 live typing requires --target-window")
            if args.capture or args.select:
                png, _ = capture_source(args.config, args.select)
                with Image.open(BytesIO(png)) as image:
                    result = engine.recognize(image)
            else:
                result = recognize_file(args.image, engine)
            result = replace(result, text=normalize_typing_text(result.text))
            if not result.text:
                raise TypingError("OCR returned no text; nothing to type")
            print(f"OCR latency: {result.latency_ms:.1f} ms", file=sys.stderr)
            if not args.live:
                print("DRY RUN: no keyboard events sent", file=sys.stderr)
                print(result.text)
                return 0
            if args.expected_file is not None:
                expected = normalize_typing_text(args.expected_file.read_text(encoding="utf-8"))
                if result.text != expected:
                    raise TypingError("OCR text differs from expected file; live typing cancelled")
            stop = Event()
            if display == "wayland":
                context = KdeWaylandInput(stop)
            else:
                context = EscapeStop(stop)
            with context as emergency_stop:
                if display == "wayland":
                    input_backend = emergency_stop
                    print("Click the authorized target window to select it.", file=sys.stderr)
                    title = input_backend.select_target()
                    print(f"Selected target: {title}", file=sys.stderr)
                    shortcut = "Meta+Shift+F12"
                else:
                    input_backend = XdotoolBackend(args.target_window)
                    shortcut = "Escape"
                print(f"Refocus the target window now; typing starts in {args.focus_wait} seconds. "
                      f"Press {shortcut} to stop.", file=sys.stderr)
                if stop.wait(args.focus_wait):
                    print("Stopped before typing", file=sys.stderr)
                    return 130
                emergency_stop.verify_running()
                input_backend.verify_target()
                outcome = TypingController(
                    input_backend, stop, typing_settings.character_delay_ms,
                    safety_check=emergency_stop.verify_running,
                ).type_text(result.text)
            print(f"Typed {outcome.typed_count} characters; "
                  f"{'stopped' if outcome.stopped else 'completed'}", file=sys.stderr)
            return 130 if outcome.stopped else 0
        except KeyboardInterrupt:
            print("Stopped by Ctrl-C", file=sys.stderr)
            return 130
        except (ConfigError, DisplayError, CaptureError, SelectionError, OcrError, TypingError, OSError) as exc:
            print(f"Error: {exc}", file=sys.stderr)
            return 1

    try:
        if args.output.exists():
            raise CaptureError(f"Output already exists: {args.output}")
        png, size = capture_source(args.config, args.select, backend)
        with args.output.open("xb") as stream:
            stream.write(png)
    except (ConfigError, DisplayError, CaptureError, SelectionError, OSError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    print(f"Saved {args.output} ({size[0]}x{size[1]})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
