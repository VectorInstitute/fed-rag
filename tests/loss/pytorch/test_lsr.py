import pytest

from fed_rag.exceptions.loss import InvalidReductionParam
from fed_rag.loss.pytorch.lsr import LSRLoss, ReductionMode


def test_lsr_loss_init() -> None:
    loss = LSRLoss(reduction="sum")

    assert loss.reduction == ReductionMode.SUM


def test_invalid_reduction_raises_error() -> None:
    with pytest.raises(InvalidReductionParam):
        LSRLoss(reduction="invalid_reduction")
