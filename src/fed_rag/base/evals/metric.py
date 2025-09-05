"""Base EvaluationMetric"""

from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel


class BaseEvaluationMetric(BaseModel, ABC):
    """Base class for evaluation metrics.

    This abstract class defines the interface for evaluation metrics that
    compare a model's prediction against the expected ground truth. Subclasses
    must implement the :meth:`__call__` method, which makes instances of the
    metric callable like a function.

    """

    @abstractmethod
    def __call__(
        self, prediction: str, actual: str, *args: Any, **kwargs: Any
    ) -> float:
        """Evaluate a prediction against the actual response.

        Args:
            prediction (str): The model's predicted output.
            actual (str): The ground-truth or expected output.
            *args (Any): Optional positional arguments for customization.
            **kwargs (Any): Optional keyword arguments for customization.

        Returns:
            float: A numerical score representing how well the prediction
            matches the actual output. The interpretation of the score
            depends on the specific metric implementation.

        Raises:
            NotImplementedError: If the subclass does not implement this method.
        """
        ...
