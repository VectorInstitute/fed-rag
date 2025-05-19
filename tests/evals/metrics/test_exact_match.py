import pytest

from fed_rag.evals import ExactMatchEvaluationMetric


@pytest.mark.parametrize(
    ("pred", "actual", "expected"),
    [
        ("1+1=2", "1+1=2", 1.0),
        ("Yes, Correct!", "yes, correct!", 1.0),
        ("not the same", "as me", 0.0),
    ],
    ids=["match", "match case insensitive", "not match"],
)
def test_exact_match(pred: str, actual: str, expected: float) -> None:
    metric = ExactMatchEvaluationMetric()

    # act
    res = metric(prediction=pred, actual=actual)

    assert res == expected
