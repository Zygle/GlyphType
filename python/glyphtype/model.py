"""Compact convolutional recurrent CTC recognizer for one text line."""

from __future__ import annotations

import torch
from torch import nn

from glyphtype.model_data import IMAGE_HEIGHT, MAX_WIDTH

BLANK = 0


class LineRecognizer(nn.Module):
    def __init__(self, classes: int, hidden_size: int = 96):
        super().__init__()
        if classes < 2:
            raise ValueError("At least one character plus CTC blank is required")
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 96, 3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 1)),
        )
        self.sequence = nn.LSTM(96, hidden_size, num_layers=1,
                                batch_first=True, bidirectional=True)
        self.classifier = nn.Linear(hidden_size * 2, classes)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        features = self.features(images).mean(dim=2).permute(0, 2, 1)
        sequence, _ = self.sequence(features)
        return self.classifier(sequence)

    @staticmethod
    def output_lengths(widths: torch.Tensor) -> torch.Tensor:
        return (widths // 4).to(dtype=torch.long)


def decode_greedy(logits: torch.Tensor, charset: str,
                  lengths: torch.Tensor | None = None) -> list[str]:
    """Collapse CTC repeats and blanks from B,T,C logits."""
    predictions = logits.argmax(dim=-1).detach().cpu().tolist()
    decoded = []
    limits = lengths.tolist() if lengths is not None else [len(row) for row in predictions]
    for sequence, limit in zip(predictions, limits):
        previous = BLANK
        chars = []
        for token in sequence[:limit]:
            if token != BLANK and token != previous:
                chars.append(charset[token - 1])
            previous = token
        decoded.append("".join(chars))
    return decoded
