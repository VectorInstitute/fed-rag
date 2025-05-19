from typing import Any

from fed_rag.base.evals.metric import BaseEvaluationMetric


class MyMetric(BaseEvaluationMetric):
    def __call__(
        self, prediction: str, actual: str, *args: Any, **kwargs: Any
    ) -> float:
        return 0.42
