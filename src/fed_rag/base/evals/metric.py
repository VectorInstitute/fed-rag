"""Base EvaluationMetric"""

from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel


class BaseEvaluationMetric(BaseModel, ABC):
    """Base Data Collator."""

    @abstractmethod
    def __call__(
        self, prediction: str, actual: str, *args: Any, **kwargs: Any
    ) -> float:
        """Evaluate an example prediction against the actual response."""
