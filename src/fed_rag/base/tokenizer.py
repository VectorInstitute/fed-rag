"""Base Tokenizer"""

from abc import ABC, abstractmethod
from typing import Any, TypedDict

from pydantic import BaseModel, ConfigDict


class EncodeResult(TypedDict):
    input_ids: list[int]
    attention_mask: list[int] | None


class BaseTokenizer(BaseModel, ABC):
    """Base Tokenizer Class."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @abstractmethod
    def encode(self, input: str, **kwargs: dict) -> EncodeResult:
        """Encode the input string into list of integers."""

    @abstractmethod
    def decode(self, input_ids: str, **kwargs: dict) -> str:
        """Decode the input token ids into a string."""

    @property
    @abstractmethod
    def unwrapped(self) -> Any:
        """Return the underlying tokenizer if there is one."""
