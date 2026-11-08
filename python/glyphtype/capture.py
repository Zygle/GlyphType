"""Display-independent capture contract and MSS X11 implementation."""

from __future__ import annotations

from typing import Protocol

from glyphtype.config import Region


class CaptureError(RuntimeError):
    """A screen region could not be captured."""


class CaptureBackend(Protocol):
    def capture_png(self, region: Region) -> bytes:
        """Capture only the requested rectangle and return PNG bytes."""


def validate_bounds(region: Region, desktop: Region) -> None:
    """Require a rectangle fully inside the X11 virtual desktop."""
    if (
        region.x < desktop.x
        or region.y < desktop.y
        or region.x + region.width > desktop.x + desktop.width
        or region.y + region.height > desktop.y + desktop.height
    ):
        raise CaptureError(
            f"Capture region {region} lies outside virtual desktop {desktop}"
        )


class X11MSSCapture:
    """Capture an X11 rectangle with MSS, opening the connection per capture."""

    def capture_png(self, region: Region) -> bytes:
        try:
            import mss
            from mss import tools
        except ImportError as exc:
            raise CaptureError("MSS is not installed; install the project package") from exc

        try:
            with mss.mss() as screen:
                virtual = screen.monitors[0]
                desktop = Region(
                    virtual["left"], virtual["top"], virtual["width"], virtual["height"]
                )
                validate_bounds(region, desktop)
                shot = screen.grab(
                    {"left": region.x, "top": region.y,
                     "width": region.width, "height": region.height}
                )
                return tools.to_png(shot.rgb, shot.size)
        except CaptureError:
            raise
        except Exception as exc:
            raise CaptureError(f"X11 capture failed: {exc}") from exc
