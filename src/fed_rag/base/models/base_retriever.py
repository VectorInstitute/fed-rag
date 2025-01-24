"""Base Retriever Model"""

from abc import ABC, abstractmethod
from typing import Any, List


class RetrieverMixin(ABC):
    """RetrieverMixin."""

    @abstractmethod
    def retrieve(self, query: str, top_k: int) -> List[dict[str, Any]]:
        """Retrieve chunks against a query.

        Args:
            query (str): _description_
            top_k (int): _description_

        Returns:
            List[dict[str, Any]]: _description_
        """
