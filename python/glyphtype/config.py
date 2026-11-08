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


def load_config(path: Path) -> Config:
    """Load and validate capture settings from a TOML file."""
    try:
        with path.open("rb") as stream:
            data = tomllib.load(stream)
    except OSError as exc:
        raise ConfigError(f"Cannot read config {path}: {exc.strerror or exc}") from exc
    except tomllib.TOMLDecodeError as exc:
        raise ConfigError(f"Invalid TOML in {path}: {exc}") from exc

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
