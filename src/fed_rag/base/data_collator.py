"""Base Data Collator"""

from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel, ConfigDict

from fed_rag import RAGSystem


class BaseDataCollator(BaseModel, ABC):
    """
    Base Data Collator.

    Abstract base class for collating input examples into batches that can
    be used by a retrieval-augmented generation (RAG) system.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)
    rag_system: RAGSystem

    @abstractmethod
    def __call__(self, features: list[dict[str, Any]], **kwargs: Any) -> Any:
        """Collate examples into a batch.

        Args:
            features (list[dict[str, Any]]): A list of feature dictionaries,
                where each dictionary represents one example.
            **kwargs (Any): Additional keyword arguments that may be used
                by specific implementations.

        Returns:
            Any: A collated batch, with format depending on the implementation.
        """
