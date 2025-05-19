"""Exact Match Metric"""

from typing import Any

from fed_rag.base.evals.metric import BaseEvaluationMetric


class ExactMatchEvaluationMetric(BaseEvaluationMetric):
    """Exact match evaluation metric class."""

    def __call__(
        self, prediction: str, actual: str, *args: Any, **kwargs: Any
    ) -> float:
        return float(prediction.lower() == actual.lower())
