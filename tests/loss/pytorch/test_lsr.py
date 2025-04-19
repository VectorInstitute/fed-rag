from unittest.mock import MagicMock, _Call, patch

import pytest
import torch
import torch.nn.functional as F
from torch.testing import assert_close

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
    retrieved_chunks: torch.Tensor,
    context: torch.Tensor,
    lm_scores: torch.Tensor,
) -> None:
    # retriever chunks probas
    scores = (retrieved_chunks * context).sum(dim=-1)
    retriever_scale = 1.0
    scores /= retriever_scale
    retriever_probas = torch.softmax(scores, dim=-1)

    # lm chunks probas
    lm_scale = 1.0
    lm_scores /= lm_scale
    lm_probas = torch.softmax(lm_scores, dim=-1)

    # kl divergence
    kl_div = F.kl_div(retriever_probas, lm_probas, reduction="none").sum(
        dim=-1
    )

    print(f"scores: {scores}")
    print(f"scores dim: {scores.shape}")

    print(f"probas: {retriever_probas}")
    print(f"probas dim: {retriever_probas.shape}")

    print(f"lm_probas: {lm_probas}")
    print(f"lm_probas dim: {lm_probas.shape}")

    print(f"kl_div: {kl_div}")
    print(f"kl_div mean: {kl_div.mean()}")

    assert retrieved_chunks.shape == (2, 3, 10)
    assert_close(
        retriever_probas.sum(dim=-1, keepdim=True),
        torch.Tensor([[1.0], [1.0]]),
    )
    assert_close(
        lm_probas.sum(dim=-1, keepdim=True), torch.Tensor([[1.0], [1.0]])
    )
