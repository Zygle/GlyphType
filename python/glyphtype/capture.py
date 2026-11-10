"""Display-independent capture contract and desktop implementations."""

from __future__ import annotations

import asyncio
from io import BytesIO
import os
from time import monotonic, sleep
from typing import Protocol

from PIL import Image

from glyphtype.config import Region


class CaptureError(RuntimeError):
    """A screen region could not be captured."""


class CaptureBackend(Protocol):
    def capture_png(self, region: Region) -> bytes:
        """Capture only the requested rectangle and return PNG bytes."""


def validate_bounds(region: Region, desktop: Region) -> None:
    """Require a rectangle fully inside a virtual desktop."""
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


class KWinWaylandCapture:
    """Capture exactly one KDE Wayland rectangle through KWin ScreenShot2.

    KWin may restrict this interface to approved desktop applications. In that
    case the caller gets a diagnostic instead of a whole-desktop fallback.
    """

    def capture_png(self, region: Region) -> bytes:
        if region.width * region.height > 100_000_000:
            raise CaptureError("Capture region is too large")
        try:
            return asyncio.run(self._capture(region))
        except CaptureError:
            raise
        except (OSError, RuntimeError) as exc:
            raise CaptureError(f"KDE Wayland region capture failed: {exc}") from exc

    async def _capture(self, region: Region) -> bytes:
        try:
            from dbus_next import Message, MessageType
            from dbus_next.aio import MessageBus
        except ImportError as exc:
            raise CaptureError("dbus-next is required for KDE Wayland capture") from exc

        if not hasattr(os, "memfd_create"):
            raise CaptureError("KDE Wayland capture requires Linux memfd_create")
        fd = os.memfd_create("glyphtype-region", flags=os.MFD_CLOEXEC)
        try:
            bus = await MessageBus(negotiate_unix_fd=True).connect()
            try:
                reply = await asyncio.wait_for(
                    bus.call(Message(
                        destination="org.kde.KWin",
                        path="/org/kde/KWin/ScreenShot2",
                        interface="org.kde.KWin.ScreenShot2",
                        member="CaptureArea",
                        signature="iiuua{sv}h",
                        body=[region.x, region.y, region.width, region.height, {}, 0],
                        unix_fds=[fd],
                    )), timeout=15,
                )
            finally:
                bus.disconnect()
            if reply.message_type == MessageType.ERROR:
                detail = reply.body[0] if reply.body else reply.error_name
                raise CaptureError(f"KWin denied region capture: {detail}")
            if not reply.body or not isinstance(reply.body[0], dict):
                raise CaptureError("KWin returned no image metadata")
            meta = {key: value.value for key, value in reply.body[0].items()}
            width, height, stride = (int(meta[key]) for key in ("width", "height", "stride"))
            fmt = int(meta["format"])
            if meta.get("type") != "raw" or fmt not in (16, 17):
                raise CaptureError(f"Unsupported KWin image format: {meta}")
            if width <= 0 or height <= 0 or width > 100_000 or height > 100_000:
                raise CaptureError(f"Invalid KWin image dimensions: {width}x{height}")
            if stride < width * 4 or stride > width * 8:
                raise CaptureError(f"Invalid KWin image stride: {stride}")
            expected = stride * height
            os.lseek(fd, 0, os.SEEK_SET)
            raw = bytearray()
            deadline = monotonic() + 15
            while len(raw) < expected:
                chunk = os.read(fd, min(1024 * 1024, expected - len(raw)))
                if not chunk:
                    if monotonic() >= deadline:
                        raise CaptureError(f"KWin returned {len(raw)} of {expected} image bytes")
                    sleep(0.01)  # KWin writes the memfd after replying with metadata.
                    continue
                raw.extend(chunk)
            mode = "RGBX" if fmt == 16 else "RGBA"
            image = Image.frombytes(mode, (width, height), bytes(raw), "raw", mode, stride)
            output = BytesIO()
            image.convert("RGB").save(output, format="PNG")
            return output.getvalue()
        except asyncio.TimeoutError as exc:
            raise CaptureError("KWin region capture timed out") from exc
        except KeyError as exc:
            raise CaptureError(f"KWin omitted image metadata: {exc}") from exc
        finally:
            os.close(fd)
