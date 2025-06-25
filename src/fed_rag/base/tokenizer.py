"""Base Tokenizer"""

from abc import ABC, abstractmethod
from typing import Any, TypedDict

from pydantic import BaseModel, ConfigDict


class EncodeResult(TypedDict):
    """Data container for tokenizer encoding results."""

    input_ids: list[int]
    attention_mask: list[int] | None


class BaseTokenizer(BaseModel, ABC):
    """Base Tokenizer Class.

    This abstract class provides the interface for creating Tokenizer objects that
    converts strings into tokens.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @abstractmethod
    def encode(self, input: str, **kwargs: Any) -> EncodeResult:
        """Encode the input string into list of integers.

        Args:
            input (str): The input string to be encoded.

        Returns:
            EncodeResult: The result of encoding.
        """

    @abstractmethod
    def decode(self, input_ids: list[int], **kwargs: Any) -> str:
        """Decode the input token ids into a string.

        Args:
            input_ids (list[int]): The token ids to be decoded back to text.

        Returns:
            str: The decoded text.
        """

    @property
    @abstractmethod
    def unwrapped(self) -> Any:
        """Return the underlying tokenizer if there is one."""
