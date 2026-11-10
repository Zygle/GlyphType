"""Conservative progress tracking for repeated OCR observations."""

from __future__ import annotations

from enum import Enum

from glyphtype.ocr import OcrResult


class State(str, Enum):
    IDLE = "IDLE"
    CAPTURING = "CAPTURING"
    RECOGNIZING = "RECOGNIZING"
    READY = "READY"
    TYPING = "TYPING"
    PAUSED = "PAUSED"
    ERROR = "ERROR"
    COMPLETED = "COMPLETED"


class TrackingError(RuntimeError):
    pass


class TextTracker:
    """Accept exact repeated text or a confirmed suffix; pause on other changes."""

    def __init__(self, minimum_confidence: float = 90.0, confirmations: int = 2):
        if not 0 <= minimum_confidence <= 100:
            raise ValueError("minimum_confidence must be from 0 to 100")
        if confirmations < 2:
            raise ValueError("confirmations must be at least 2")
        self.minimum_confidence = minimum_confidence
        self.confirmations = confirmations
        self.state = State.IDLE
        self.text: str | None = None
        self.typed_count = 0
        self.reason: str | None = None
        self._candidate: str | None = None
        self._candidate_count = 0

    def begin_capture(self) -> None:
        if self.state not in (State.IDLE, State.READY, State.COMPLETED):
            raise TrackingError(f"Cannot capture from {self.state.value}")
        self.state = State.CAPTURING

    def begin_recognition(self) -> None:
        if self.state != State.CAPTURING:
            raise TrackingError(f"Cannot recognize from {self.state.value}")
        self.state = State.RECOGNIZING

    def observe(self, result: OcrResult) -> None:
        if self.state != State.RECOGNIZING:
            raise TrackingError(f"Cannot observe from {self.state.value}")
        if not result.text or result.confidence is None or result.confidence < self.minimum_confidence:
            self.pause("OCR text or confidence is uncertain")
            return
        observed = result.text
        if self.text == observed:
            self._candidate = None
            self._candidate_count = 0
            self.state = State.READY if self.typed_count < len(self.text) else State.COMPLETED
            return
        if self.text is not None and not observed.startswith(self.text):
            self.pause("Displayed text changed; review before starting a new passage")
            return
        if observed == self._candidate:
            self._candidate_count += 1
        else:
            self._candidate = observed
            self._candidate_count = 1
        if self._candidate_count >= self.confirmations:
            self.text = observed
            self._candidate = None
            self._candidate_count = 0
            self.state = State.READY
        else:
            self.state = State.IDLE if self.text is None else State.COMPLETED

    def pending_text(self) -> str:
        if self.state != State.READY or self.text is None:
            raise TrackingError(f"No text ready in {self.state.value}")
        self.state = State.TYPING
        return self.text[self.typed_count:]

    def commit(self, count: int) -> None:
        if self.state != State.TYPING or self.text is None:
            raise TrackingError(f"Cannot commit from {self.state.value}")
        if not 0 <= count <= len(self.text) - self.typed_count:
            raise TrackingError("Typed count exceeds pending text")
        self.typed_count += count
        self.state = State.COMPLETED if self.typed_count == len(self.text) else State.READY

    def pause(self, reason: str) -> None:
        self.reason = reason
        self._candidate = None
        self._candidate_count = 0
        self.state = State.PAUSED

    def resume(self) -> None:
        if self.state != State.PAUSED:
            raise TrackingError("Tracker is not paused")
        self.reason = None
        if self.text is None:
            self.state = State.IDLE
        else:
            self.state = State.READY if self.typed_count < len(self.text) else State.COMPLETED

    def accept_new_text(self, text: str) -> None:
        """Explicit manual reset after reviewing a changed passage."""
        if self.state != State.PAUSED or not text:
            raise TrackingError("A paused tracker and nonempty reviewed text are required")
        self.text = text
        self.typed_count = 0
        self.reason = None
        self.state = State.READY

    def fail(self, reason: str) -> None:
        self.reason = reason
        self.state = State.ERROR
