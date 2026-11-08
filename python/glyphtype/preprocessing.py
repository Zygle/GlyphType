"""Optional, deterministic image transforms for OCR comparisons."""

from __future__ import annotations

from PIL import Image, ImageEnhance, ImageFilter

from glyphtype.config import PreprocessConfig


def preprocess(image: Image.Image, settings: PreprocessConfig) -> Image.Image:
    """Apply configured transforms to a copy; defaults preserve image pixels."""
    result = image.copy()
    if settings.grayscale:
        result = result.convert("L")
    if settings.contrast != 1.0:
        result = ImageEnhance.Contrast(result).enhance(settings.contrast)
    if settings.scale != 1.0:
        size = (max(1, round(result.width * settings.scale)),
                max(1, round(result.height * settings.scale)))
        result = result.resize(size, Image.Resampling.LANCZOS)
    if settings.median_filter:
        result = result.filter(ImageFilter.MedianFilter(size=3))
    if settings.threshold >= 0:
        result = result.convert("L").point(lambda value: 255 if value > settings.threshold else 0)
    if settings.sharpen:
        result = result.filter(ImageFilter.UnsharpMask())
    return result
