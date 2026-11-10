"""Conservative, opt-in X11 typing into a selected window."""

from __future__ import annotations

from dataclasses import dataclass
import re
import shutil
import subprocess
from threading import Event
from typing import Callable, Protocol


class TypingError(RuntimeError):
    """Typing was unsafe or an input backend failed."""


class InputBackend(Protocol):
    def verify_target(self) -> None: ...
    def send_character(self, character: str) -> None: ...


@dataclass(frozen=True)
class TypingResult:
    typed_count: int
    stopped: bool


def normalize_typing_text(text: str) -> str:
    """Treat visual OCR line wraps as spaces in a single typing passage."""
    lines = text.replace("\r\n", "\n").replace("\r", "\n")
    return re.sub(r"[ \t]*\n+[ \t]*", " ", lines).strip()


class XdotoolBackend:
    """Send X11 events to a fixed window after verifying focus."""

    def __init__(self, window_id: int):
        if window_id <= 0:
            raise TypingError("Target window ID must be positive")
        if shutil.which("xdotool") is None:
            raise TypingError("xdotool is required for live X11 typing")
        self.window_id = window_id

    @staticmethod
    def _run(*args: str) -> str:
        try:
            completed = subprocess.run(
                ["xdotool", *args], capture_output=True, text=True,
                timeout=5, check=False,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise TypingError(f"xdotool failed: {exc}") from exc
        if completed.returncode:
            raise TypingError(f"xdotool failed: {completed.stderr.strip() or completed.returncode}")
        return completed.stdout.strip()

    def verify_target(self) -> None:
        active = self._run("getactivewindow")
        try:
            active_id = int(active, 0)
        except ValueError as exc:
            raise TypingError(f"Unexpected active window ID: {active!r}") from exc
        if active_id != self.window_id:
            raise TypingError("Target window lost focus; typing stopped")
    def send_character(self, character: str) -> None:
        if character in "\r\n":
            raise TypingError("Line breaks cannot be sent as keyboard input")
        self._run(
            "type", "--window", str(self.window_id), "--clearmodifiers",
            "--delay", "0", "--", character,
        )


class TypingController:
    """Track one static passage within one explicit start action."""

    def __init__(self, backend: InputBackend, stop: Event, delay_ms: int,
                 safety_check: Callable[[], None] | None = None,
                 on_sent: Callable[[int], None] | None = None):
        self.backend = backend
        self.stop = stop
        self.delay_ms = delay_ms
        self.safety_check = safety_check
        self.on_sent = on_sent

    def type_text(self, text: str) -> TypingResult:
        if not text:
            raise TypingError("OCR returned no text; nothing to type")
        if any(ord(char) < 32 and char not in "\n\r" for char in text):
            raise TypingError("OCR text contains unsupported control characters")
        normalized = normalize_typing_text(text)
        if not normalized:
            raise TypingError("OCR returned only whitespace; nothing to type")
        count = 0
        for index, character in enumerate(normalized):
            if self.stop.is_set():
                return TypingResult(count, True)
            if self.safety_check is not None:
                self.safety_check()
            self.backend.verify_target()
            if self.stop.is_set():
                return TypingResult(count, True)
            self.backend.send_character(character)
            count += 1
            if self.on_sent is not None:
                self.on_sent(count)
            if index + 1 < len(normalized) and self.stop.wait(self.delay_ms / 1000):
                return TypingResult(count, True)
        return TypingResult(count, False)


class EscapeStop:
    """Observe Escape globally on X11; also allow Ctrl-C via the CLI."""

    def __init__(self, event: Event, retry: Event | None = None,
                 accept: Event | None = None):
        self.event = event
        self.retry = retry
        self.accept = accept
        self.listener = None

    def __enter__(self) -> "EscapeStop":
        try:
            from pynput import keyboard
            def on_press(key: object) -> None:
                if key == keyboard.Key.esc:
                    self.event.set()
                elif key == keyboard.Key.f8 and self.retry is not None:
                    self.retry.set()
                elif key == keyboard.Key.f9 and self.accept is not None:
                    self.accept.set()
            self.listener = keyboard.Listener(on_press=on_press)
            self.listener.start()
            self.listener.wait()
        except Exception as exc:
            raise TypingError(f"Cannot start X11 Escape listener: {exc}") from exc
        return self

    def verify_running(self) -> None:
        if self.listener is None or not self.listener.running:
            raise TypingError("Escape listener stopped; typing halted")

    def __exit__(self, *_: object) -> None:
        if self.listener is not None:
            self.listener.stop()
