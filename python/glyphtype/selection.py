"""Interactive X11 rectangle selection without capturing screen pixels."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from tempfile import TemporaryDirectory

from glyphtype.config import ConfigError, Region


class SelectionError(RuntimeError):
    """A rectangle was not selected."""


def select_x11_region() -> Region:
    """Ask slop for one dragged rectangle in virtual-desktop coordinates."""
    if shutil.which("slop") is None:
        raise SelectionError("slop is required for interactive X11 region selection")
    try:
        result = subprocess.run(
            ["slop", "--tolerance", "0", "--format", "%x %y %w %h"],
            capture_output=True, text=True, check=False,
        )
    except OSError as exc:
        raise SelectionError(f"Cannot start slop: {exc}") from exc
    if result.returncode:
        raise SelectionError(f"Selection cancelled or failed: {result.stderr.strip() or result.returncode}")
    parts = result.stdout.split()
    if len(parts) != 4:
        raise SelectionError(f"Unexpected selection geometry: {result.stdout.strip()!r}")
    try:
        return Region(*(int(part) for part in parts))
    except (ValueError, ConfigError) as exc:
        raise SelectionError(f"Invalid selection geometry: {result.stdout.strip()!r}") from exc


def capture_wayland_selection() -> bytes:
    """Let KDE Spectacle select a rectangle and return only its cropped PNG."""
    if shutil.which("spectacle") is None:
        raise SelectionError("KDE Spectacle is required for Wayland drag selection")
    with TemporaryDirectory(prefix="glyphtype-select-") as directory:
        output = Path(directory) / "selection.png"
        try:
            result = subprocess.run(
                ["spectacle", "--region", "--background", "--output", str(output)],
                capture_output=True, text=True, timeout=120, check=False,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise SelectionError(f"Wayland selection failed: {exc}") from exc
        if result.returncode or not output.is_file():
            raise SelectionError(
                f"Wayland selection cancelled or failed: {result.stderr.strip() or result.returncode}"
            )
        return output.read_bytes()
