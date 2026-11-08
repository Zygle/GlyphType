"""Small accuracy metrics for labeled OCR samples."""

from __future__ import annotations


def character_error_rate(expected: str, actual: str) -> float:
    """Levenshtein distance divided by reference character count."""
    if not expected:
        return 0.0 if not actual else 1.0
    previous = list(range(len(actual) + 1))
    for row, reference_char in enumerate(expected, start=1):
        current = [row]
        for column, actual_char in enumerate(actual, start=1):
            current.append(min(
                current[column - 1] + 1,
                previous[column] + 1,
                previous[column - 1] + (reference_char != actual_char),
            ))
        previous = current
    return previous[-1] / len(expected)
