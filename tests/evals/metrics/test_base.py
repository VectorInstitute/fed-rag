from typing import Any

from fed_rag.base.evals.metric import BaseEvaluationMetric


class MyMetric(BaseEvaluationMetric):
    def __call__(
        self, prediction: str, actual: str, *args: Any, **kwargs: Any
    ) -> float:
        return 0.42


def test_metric_call() -> None:
    metric = MyMetric()

    score = metric("fake pred", "fake actual")

    assert score == 0.42
