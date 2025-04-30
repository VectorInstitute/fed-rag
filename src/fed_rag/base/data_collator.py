"""Base Data Collator"""

from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel, ConfigDict

from fed_rag.types.rag_system import RAGSystem


class BaseDataCollator(BaseModel, ABC):
    """Base Data Collator."""

    model_config = ConfigDict(arbitrary_types_allowed=True)
    rag_system: RAGSystem

    @abstractmethod
    def __call__(self, features: list[dict[str, Any]], **kwargs: Any) -> Any:
        """Collate examples into a batch."""
