"""PyTorch RAG Finetuning Dataset"""

from typing import Any

import torch
from torch.utils.data import Dataset


class PyTorchRAGFinetuningDataset(Dataset):
    """PyTorch RAG Fine-Tuning Dataset Class.

    Args:
        Dataset (_type_): _description_
    """

    def __init__(
        self, input_ids: list[torch.Tensor], target_ids: list[torch.Tensor]
    ):
        self.input_ids = input_ids
        self.target_ids = target_ids

    def __len__(self) -> int:
        return len(self.input_ids)

    def __getitem__(self, idx: int) -> Any:
        return self.input_ids[idx], self.target_ids[idx]
