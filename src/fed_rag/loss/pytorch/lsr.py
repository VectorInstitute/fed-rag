"""LM-Supervised Retriever Loss."""

from enum import Enum

import torch
import torch.functional as F
import torch.nn as nn
from typing_extensions import assert_never


class ReductionMode(str, Enum):
    MEAN = "mean"
    SUM = "sum"


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
        self.reduction = reduction

    def forward(
        self, retrieval_logits: torch.Tensor, lm_logits: torch.Tensor
    ) -> torch.Tensor:
        retrieval_probs = F.softmax(retrieval_logits, dim=1)
        lm_probs = F.softmax(lm_logits, dim=1)

        kl_div = F.kl_div(retrieval_probs, lm_probs, reduction="none").sum(
            dim=1
        )

        match self.reduction:
            case ReductionMode.MEAN:
                return kl_div.mean()
            case ReductionMode.SUM:
                return kl_div.sum()
            case _:
                assert_never(self.reduction)  # pragma: no cover
