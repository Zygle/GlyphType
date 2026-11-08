"""Display-server detection and capture support diagnostics."""

from __future__ import annotations

from collections.abc import Mapping
import os


class DisplayError(RuntimeError):
    """The active display cannot support the requested capture backend."""


def detect_display(env: Mapping[str, str] | None = None) -> str:
    """Return x11, wayland, or unknown; prefer the session declaration."""
    values = os.environ if env is None else env
    session = values.get("XDG_SESSION_TYPE", "").strip().lower()
    if session in ("x11", "wayland"):
        return session
    if values.get("WAYLAND_DISPLAY"):
        return "wayland"
    if values.get("DISPLAY"):
        return "x11"
    return "unknown"


def require_x11(backend: str, env: Mapping[str, str] | None = None) -> None:
    """Reject unsupported or ambiguous sessions before opening X11 capture."""
    display = detect_display(env)
    if display == "wayland":
        raise DisplayError(
            "Wayland session detected. Phase 1 captures only native X11 desktops; "
            "Xwayland does not expose the full Wayland desktop. Log into an X11 session."
        )
    if display != "x11":
        raise DisplayError(
            "No supported display server detected. Start an X11 desktop session "
            "with XDG_SESSION_TYPE=x11 and a valid DISPLAY."
        )
    values = os.environ if env is None else env
    if not values.get("DISPLAY"):
        raise DisplayError("X11 session detected, but DISPLAY is unset")
    if backend not in ("auto", "x11"):
        raise DisplayError(f"Unsupported capture backend: {backend}")
