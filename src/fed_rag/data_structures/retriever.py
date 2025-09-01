"""Data structures for retrievers."""

from typing import TypedDict

import torch


class EncodeResult(TypedDict):
    """
    Represents the result of encoding multiple types of data.

    This TypedDict is used as a structured output for encoding operations
    involving various data modalities such as text, image, audio, or video.
    Each key corresponds to a specific modality and may contain a tensor
    result or None if that modality is not used or applicable.

    Attributes:
    text: Union[torch.Tensor, None]
        The tensor representation of encoded text data, or None if text
        is not processed.
    image: Union[torch.Tensor, None]
        The tensor representation of encoded image data, or None if image
        processing is not performed.
    audio: Union[torch.Tensor, None]
        The tensor representation of encoded audio data, or None if audio
        is not processed.
    video: Union[torch.Tensor, None]
        The tensor representation of encoded video data, or None if video
        processing is not performed.
    """

    text: torch.Tensor | None
    image: torch.Tensor | None
    audio: torch.Tensor | None
    video: torch.Tensor | None
