"""Opt-in KDE Wayland input through the RemoteDesktop portal.

KWin reports the active native window to a temporary D-Bus service. The
portal supplies keyboard access only after the desktop grants a session.
"""

from __future__ import annotations

import asyncio
from concurrent.futures import Future, TimeoutError as FutureTimeout
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Event, Thread
from uuid import uuid4

from glyphtype.typing import TypingError


class KdeWaylandInput:
    """Maintain a portal session, target identity, and KWin stop shortcut."""

    def __init__(self, stop: Event,
                 retry: Event | None = None, accept: Event | None = None):
        self.stop = stop
        self.retry = retry
        self.accept = accept
        self.shortcut_ready: bool | None = None
        self.target_uuid: str | None = None
        self.active_uuid: str | None = None
        self.loop = asyncio.new_event_loop()
        self.thread = Thread(target=self._run_loop, daemon=True)
        self.bus = None
        self.session: str | None = None
        self.script_name = f"glyphtype_{os.getpid()}_{uuid4().hex[:8]}"
        self.service_name = f"org.glyphtype.Focus.p{os.getpid()}.t{uuid4().hex[:8]}"
        self.directory: TemporaryDirectory[str] | None = None
        self._response_waiters: dict[str, asyncio.Future] = {}
        self._response_cache: dict[str, tuple[int, dict]] = {}

    def _run_loop(self) -> None:
        asyncio.set_event_loop(self.loop)
        self.loop.run_forever()

    def _call(self, coroutine, timeout: float = 30):
        future: Future = asyncio.run_coroutine_threadsafe(coroutine, self.loop)
        try:
            return future.result(timeout)
        except FutureTimeout as exc:
            future.cancel()
            raise TypingError("Wayland desktop operation timed out") from exc
        except TypingError:
            future.cancel()
            raise
        except Exception as exc:
            future.cancel()
            raise TypingError(f"Wayland desktop operation failed: {exc}") from exc

    def __enter__(self) -> "KdeWaylandInput":
        self.thread.start()
        try:
            self._call(self._setup_focus(), 15)
            self._call(self._setup_portal(), 150)
            return self
        except Exception:
            self.__exit__(None, None, None)
            raise

    def __exit__(self, *_: object) -> None:
        try:
            if self.thread.is_alive():
                try:
                    self._call(self._close(), 10)
                finally:
                    self.loop.call_soon_threadsafe(self.loop.stop)
                    self.thread.join(timeout=2)
                    if not self.thread.is_alive():
                        self.loop.close()
        finally:
            if self.directory is not None:
                self.directory.cleanup()
                self.directory = None

    async def _dbus_call(self, destination: str, path: str, interface: str,
                         member: str, signature: str = "", body: list | None = None):
        from dbus_next import Message, MessageType

        reply = await self.bus.call(Message(
            destination=destination, path=path, interface=interface,
            member=member, signature=signature, body=body or [],
        ))
        if reply.message_type == MessageType.ERROR:
            detail = reply.body[0] if reply.body else reply.error_name
            raise TypingError(f"{member} failed: {detail}")
        return reply.body

    def _handle_message(self, message) -> bool:
        from dbus_next import Message, MessageType

        if (message.message_type == MessageType.METHOD_CALL
                and message.path == "/org/glyphtype/Focus"
                and message.interface == "org.glyphtype.Focus"):
            if message.member == "Update" and message.signature == "s":
                self.active_uuid = message.body[0]
            elif message.member == "Stop" and message.signature == "":
                self.stop.set()
            elif message.member == "Retry" and message.signature == "" and self.retry is not None:
                self.retry.set()
            elif message.member == "Accept" and message.signature == "" and self.accept is not None:
                self.accept.set()
            elif message.member == "ShortcutReady" and message.signature == "b":
                self.shortcut_ready = bool(message.body[0])
            else:
                self.bus.send(Message.new_error(message, "org.glyphtype.InvalidCall", "Unknown focus call"))
                return True
            self.bus.send(Message.new_method_return(message))
            return True
        if (message.message_type == MessageType.SIGNAL
                and message.interface == "org.freedesktop.portal.Request"
                and message.member == "Response"):
            code = int(message.body[0])
            result = {key: value.value for key, value in message.body[1].items()}
            waiter = self._response_waiters.pop(message.path, None)
            if waiter is None:
                self._response_cache[message.path] = (code, result)
            elif not waiter.done():
                waiter.set_result((code, result))
        return False

    async def _setup_focus(self) -> None:
        from dbus_next.aio import MessageBus

        self.bus = await MessageBus().connect()
        self.bus.add_message_handler(self._handle_message)
        await self.bus.request_name(self.service_name)
        self.directory = TemporaryDirectory(prefix="glyphtype-kwin-")
        script_path = Path(self.directory.name) / "focus.js"
        service = json.dumps(self.service_name)
        retry_script = (
            f"registerShortcut({json.dumps(self.script_name + '_retry')}, 'Retry GlyphType OCR',\n"
            "  'Meta+Shift+F8', function() {\n"
            f"    callDBus({service}, '/org/glyphtype/Focus', 'org.glyphtype.Focus', 'Retry');\n"
            "  });\n"
        ) if self.retry is not None else ""
        accept_script = (
            f"registerShortcut({json.dumps(self.script_name + '_accept')}, 'Accept GlyphType passage',\n"
            "  'Meta+Shift+F9', function() {\n"
            f"    callDBus({service}, '/org/glyphtype/Focus', 'org.glyphtype.Focus', 'Accept');\n"
            "  });\n"
        ) if self.accept is not None else ""
        script_path.write_text(
            "function report() {\n"
            "  const w = workspace.activeWindow;\n"
            f"  callDBus({service}, '/org/glyphtype/Focus', 'org.glyphtype.Focus',\n"
            "    'Update', w ? String(w.internalId) : '');\n"
            "}\n"
            "workspace.windowActivated.connect(report);\n"
            f"const stopReady = registerShortcut({json.dumps(self.script_name)}, 'Stop GlyphType typing',\n"
            "  'Meta+Shift+F12', function() {\n"
            f"    callDBus({service}, '/org/glyphtype/Focus', 'org.glyphtype.Focus', 'Stop');\n"
            "  });\n"
            f"callDBus({service}, '/org/glyphtype/Focus', 'org.glyphtype.Focus', 'ShortcutReady', stopReady);\n"
            + retry_script + accept_script +
            "report();\n", encoding="utf-8",
        )
        await self._dbus_call("org.kde.KWin", "/Scripting", "org.kde.kwin.Scripting",
                              "loadScript", "ss", [str(script_path), self.script_name])
        await self._dbus_call("org.kde.KWin", "/Scripting", "org.kde.kwin.Scripting", "start")
        for _ in range(50):
            if self.active_uuid is not None and self.shortcut_ready is not None:
                if not self.shortcut_ready:
                    raise TypingError("KWin could not register the emergency stop shortcut")
                return
            await asyncio.sleep(0.1)
        raise TypingError("KWin did not report the active window; live input cancelled")

    async def _portal_request(self, member: str, signature: str, body: list,
                              token: str) -> dict:
        from dbus_next import Variant

        options = {"handle_token": Variant("s", token)}
        body[-1].update(options)
        reply = await self._dbus_call(
            "org.freedesktop.portal.Desktop", "/org/freedesktop/portal/desktop",
            "org.freedesktop.portal.RemoteDesktop", member, signature, body,
        )
        handle = reply[0]
        cached = self._response_cache.pop(handle, None)
        if cached is None:
            waiter = asyncio.get_running_loop().create_future()
            self._response_waiters[handle] = waiter
            try:
                code, result = await asyncio.wait_for(waiter, 120)
            finally:
                self._response_waiters.pop(handle, None)
        else:
            code, result = cached
        if code != 0:
            raise TypingError(f"Wayland desktop denied or cancelled {member} (response {code})")
        return result

    async def _setup_portal(self) -> None:
        from dbus_next import Variant

        await self._dbus_call(
            "org.freedesktop.DBus", "/org/freedesktop/DBus", "org.freedesktop.DBus",
            "AddMatch", "s", ["type='signal',interface='org.freedesktop.portal.Request',member='Response'"],
        )
        token = uuid4().hex
        result = await self._portal_request("CreateSession", "a{sv}", [
            {"session_handle_token": Variant("s", token)}], uuid4().hex)
        self.session = result.get("session_handle")
        if not self.session:
            raise TypingError("RemoteDesktop portal returned no session")
        await self._portal_request("SelectDevices", "oa{sv}", [
            self.session, {"types": Variant("u", 1)}], uuid4().hex)
        result = await self._portal_request("Start", "osa{sv}", [
            self.session, "", {}], uuid4().hex)
        if not int(result.get("devices", 0)) & 1:
            raise TypingError("RemoteDesktop portal did not grant keyboard input")

    async def _select_target(self) -> tuple[str, str]:
        reply = await self._dbus_call(
            "org.kde.KWin", "/KWin", "org.kde.KWin", "queryWindowInfo")
        info = {key: value.value for key, value in reply[0].items()}
        identity = str(info.get("uuid", ""))
        title = str(info.get("caption", ""))
        if not identity:
            raise TypingError("No application window was selected")
        return identity, title

    def select_target(self) -> str:
        """Ask the user to click the authorized native target window."""
        self.target_uuid, title = self._call(self._select_target(), 120)
        return title

    def verify_target(self) -> None:
        if self.stop.is_set():
            raise TypingError("Stopped by emergency shortcut")
        self.verify_running()
        if not self.target_uuid or self.active_uuid != self.target_uuid:
            raise TypingError("Selected Wayland window lost focus; typing stopped")

    def verify_running(self) -> None:
        if not self.thread.is_alive() or self.bus is None or not self.bus.connected:
            raise TypingError("Wayland focus monitor disconnected; typing stopped")

    async def _send_keysym(self, symbol: int, state: int) -> None:
        await self._dbus_call(
            "org.freedesktop.portal.Desktop", "/org/freedesktop/portal/desktop",
            "org.freedesktop.portal.RemoteDesktop", "NotifyKeyboardKeysym", "oa{sv}iu",
            [self.session, {}, symbol, state],
        )

    def send_character(self, character: str) -> None:
        self.verify_target()
        if len(character) != 1:
            raise TypingError("Keyboard backend expects one character")
        if character in "\r\n":
            raise TypingError("Line breaks cannot be sent as keyboard input")
        symbol = ord(character)
        if symbol > 0xFF and symbol < 0x01000000:
            symbol |= 0x01000000  # XKB Unicode keysym
        self._call(self._send_keysym(symbol, 1), 5)
        try:
            self._call(self._send_keysym(symbol, 0), 5)
        except Exception as exc:
            raise TypingError(f"Key release failed; stop and inspect the target: {exc}") from exc

    async def _close(self) -> None:
        if self.bus is None:
            return
        if self.session:
            try:
                await self._dbus_call(
                    "org.freedesktop.portal.Desktop", self.session,
                    "org.freedesktop.portal.Session", "Close")
            except TypingError:
                pass  # Desktop session may already have been revoked.
        try:
            await self._dbus_call("org.kde.KWin", "/Scripting", "org.kde.kwin.Scripting",
                                  "unloadScript", "s", [self.script_name])
        finally:
            self.bus.disconnect()
