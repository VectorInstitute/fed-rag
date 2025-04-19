from unittest.mock import MagicMock, _Call, patch

import pytest
import torch

from fed_rag.exceptions.loss import InvalidReductionParam
from fed_rag.loss.pytorch.lsr import LSRLoss, ReductionMode


def test_lsr_loss_init() -> None:
    loss = LSRLoss(reduction="sum")

    assert loss.reduction == ReductionMode.SUM


def test_invalid_reduction_raises_error() -> None:
    with pytest.raises(InvalidReductionParam):
        LSRLoss(reduction="invalid_reduction")


@pytest.mark.parametrize(
    ("reduction", "expected"), [("mean", 10.5), ("sum", 21)]
)
@patch("fed_rag.loss.pytorch.lsr.F")
def test_lsr_forward(
    mock_torch_functional: MagicMock, reduction: str, expected: float
) -> None:
    # arrange mocks
    mock_torch_functional.softmax.side_effect = iter(
        [torch.Tensor([1, 2, 3]), torch.Tensor([4, 5, 6])]
    )
    mock_torch_functional.kl_div.return_value = torch.Tensor(
        [[1, 2, 3], [4, 5, 6]]
    )

    loss = LSRLoss(reduction=reduction)
    retrieval_logits = torch.zeros(3)
    lm_logits = torch.zeros(3)
    out = loss(retrieval_logits, lm_logits)

    # assert
    calls = [
        _Call(((retrieval_logits,), {"dim": 1})),
        _Call(((lm_logits,), {"dim": 1})),
    ]
    mock_torch_functional.softmax.assert_has_calls(calls)
    mock_torch_functional.kl_div.assert_called_once()
    assert out == torch.Tensor([expected])


def test_lsr_forward_2(
    retrieved_chunks: torch.Tensor, context: torch.Tensor
) -> None:
    scores = retrieved_chunks * context
    scores = scores.sum(dim=-1).unsqueeze(-1)

    print(f"scores: {scores}")
    print(f"scores dim: {scores.shape}")

    pytest.fail()
    assert retrieved_chunks.shape == (2, 3, 10)
