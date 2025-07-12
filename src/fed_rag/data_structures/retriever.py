"""Data structures for retrievers."""

from typing import TypedDict

import torch


class EncodeResult(TypedDict):
    """Result of encoder."""

    text: torch.Tensor | None
    image: torch.Tensor | None
    audio: torch.Tensor | None
    video: torch.Tensor | None
