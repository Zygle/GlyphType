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
    """Validate the optional legacy X11 backend."""
    display = detect_display(env)
    if display != "x11":
        raise DisplayError(f"X11 backend requires an X11 session; detected {display}")
    values = os.environ if env is None else env
    if not values.get("DISPLAY"):
        raise DisplayError("X11 session detected, but DISPLAY is unset")
    if backend not in ("auto", "x11"):
        raise DisplayError(f"Unsupported capture backend: {backend}")


def resolve_backend(backend: str, env: Mapping[str, str] | None = None) -> str:
    """Choose the native backend and reject mismatched display settings."""
    values = os.environ if env is None else env
    display = detect_display(values)
    selected = display if backend == "auto" else backend
    if selected not in ("x11", "wayland"):
        raise DisplayError(f"No supported display server detected ({display})")
    if selected != display:
        raise DisplayError(f"Configured {selected} backend, but active session is {display}")
    if selected == "x11":
        require_x11(backend, values)
    elif not values.get("WAYLAND_DISPLAY"):
        raise DisplayError("Wayland session detected, but WAYLAND_DISPLAY is unset")
    elif "KDE" not in values.get("XDG_CURRENT_DESKTOP", "").upper():
        raise DisplayError("Native Wayland capture currently supports KDE Plasma only")
    return selected
