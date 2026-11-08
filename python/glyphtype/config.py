"""TOML configuration for the capture prototype."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import tomllib


class ConfigError(ValueError):
    """A configuration file is missing or invalid."""


@dataclass(frozen=True)
class Region:
    x: int
    y: int
    width: int
    height: int

    def __post_init__(self) -> None:
        for name in ("x", "y", "width", "height"):
            if type(getattr(self, name)) is not int:
                raise ConfigError(f"capture.{name} must be an integer")
        if self.width <= 0 or self.height <= 0:
            raise ConfigError("capture.width and capture.height must be positive")


@dataclass(frozen=True)
class Config:
    backend: str
    region: Region


@dataclass(frozen=True)
class PreprocessConfig:
    grayscale: bool = False
    contrast: float = 1.0
    scale: float = 1.0
    threshold: int = -1
    median_filter: bool = False
    sharpen: bool = False


@dataclass(frozen=True)
class OcrConfig:
    language: str = "eng"
    executable: str = "tesseract"
    page_segmentation_mode: int = 6
    preprocessing: PreprocessConfig = PreprocessConfig()


def _read_toml(path: Path) -> dict:
    try:
        with path.open("rb") as stream:
            return tomllib.load(stream)
    except OSError as exc:
        raise ConfigError(f"Cannot read config {path}: {exc.strerror or exc}") from exc
    except tomllib.TOMLDecodeError as exc:
        raise ConfigError(f"Invalid TOML in {path}: {exc}") from exc


def load_config(path: Path) -> Config:
    """Load and validate capture settings from a TOML file."""
    data = _read_toml(path)

    display = data.get("display", {})
    capture = data.get("capture")
    if not isinstance(display, dict):
        raise ConfigError("[display] must be a table")
    if not isinstance(capture, dict):
        raise ConfigError("[capture] table is required")
    backend = display.get("backend", "auto")
    if backend not in ("auto", "x11"):
        raise ConfigError("display.backend must be 'auto' or 'x11'")
    try:
        region = Region(*(capture[name] for name in ("x", "y", "width", "height")))
    except KeyError as exc:
        raise ConfigError(f"capture.{exc.args[0]} is required") from exc
    return Config(backend=backend, region=region)


def load_ocr_config(path: Path | None = None) -> OcrConfig:
    """Read OCR settings without requiring capture coordinates."""
    data = _read_toml(path) if path is not None else {}
    ocr = data.get("ocr", {})
    prep = data.get("preprocessing", {})
    if not isinstance(ocr, dict) or not isinstance(prep, dict):
        raise ConfigError("[ocr] and [preprocessing] must be tables")
    language = ocr.get("language", "eng")
    executable = ocr.get("executable", "tesseract")
    psm = ocr.get("page_segmentation_mode", 6)
    if not isinstance(language, str) or not language or any(c.isspace() for c in language):
        raise ConfigError("ocr.language must be a nonempty language code")
    if not isinstance(executable, str) or not executable:
        raise ConfigError("ocr.executable must be a nonempty path or command")
    if type(psm) is not int or not 0 <= psm <= 13:
        raise ConfigError("ocr.page_segmentation_mode must be an integer from 0 to 13")

    defaults = PreprocessConfig()
    values = {name: prep.get(name, getattr(defaults, name)) for name in
              ("grayscale", "contrast", "scale", "threshold", "median_filter", "sharpen")}
    for name in ("grayscale", "median_filter", "sharpen"):
        if type(values[name]) is not bool:
            raise ConfigError(f"preprocessing.{name} must be a boolean")
    for name in ("contrast", "scale"):
        if type(values[name]) not in (int, float) or values[name] <= 0:
            raise ConfigError(f"preprocessing.{name} must be a positive number")
        values[name] = float(values[name])
    if type(values["threshold"]) is not int or not -1 <= values["threshold"] <= 255:
        raise ConfigError("preprocessing.threshold must be -1 or an integer from 0 to 255")
    return OcrConfig(language, executable, psm, PreprocessConfig(**values))
