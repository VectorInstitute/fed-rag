"""LM-Supervised Retriever Loss."""

from enum import Enum

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing_extensions import assert_never

from fed_rag.exceptions.loss import InvalidReductionParam


class ReductionMode(str, Enum):
    """Reduction mode enum."""

    MEAN = "mean"
    SUM = "sum"

    @classmethod
    def members_list(cls) -> list[str]:
        return [member for member in cls]


class LSRLoss(nn.Module):
    """PyTorch implementation of the LM-Supervised Retriever Loss.

    Given input context x and ground truth continuation y, computes KL divergence
    between retrieval likelihood P_R(d|x) and language model likelihood Q_LM(d|x,y),
    where d is the retrieved document.

    Source: Shi, Weijia, et al. "Replug: Retrieval-augmented black-box language models."
        arXiv preprint arXiv:2301.12652 (2023).
    Arxiv: https://arxiv.org/pdf/2301.12652
    """

    def __init__(self, reduction: ReductionMode = ReductionMode.MEAN):
        # This line is critical - it initializes all the Module machinery
        super(LSRLoss, self).__init__()

        if reduction not in ReductionMode.members_list():
            msg = (
                f"Invalid reduction {reduction}. "
                f"Valid reductions are: {', '.join(ReductionMode.members_list())}"
            )
            raise InvalidReductionParam(msg)

        self.reduction = reduction

    def forward(
        self, retrieval_scores: torch.Tensor, lm_scores: torch.Tensor
    ) -> torch.Tensor:
        retrieval_probs = F.softmax(retrieval_scores, dim=1)
        lm_probs = F.softmax(lm_scores, dim=1)
        kl_div = F.kl_div(retrieval_probs, lm_probs, reduction="none").sum(
            dim=-1
        )

        match self.reduction:
            case ReductionMode.MEAN:
                return kl_div.mean()
            case ReductionMode.SUM:
                return kl_div.sum()
            case _:  # pragma: no cover
                assert_never(self.reduction)  # pragma: no cover
