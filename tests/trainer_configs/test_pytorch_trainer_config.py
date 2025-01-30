"""PyTorchTrainerConfig Unit Tests"""

from typing import Any

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, Dataset

from fed_rag.trainer_configs import PyTorchTrainerConfig


class _TestDataset(Dataset):
    def __init__(self, size: int) -> None:
        self.features = np.random.rand(size, 2)
        self.labels = np.random.choice(2, size=size)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> tuple[np.ndarray, Any]:
        return self.features[index], self.labels[index]


@pytest.fixture()
def train_dataloader() -> DataLoader:
    dataset = _TestDataset(size=10)
    return DataLoader(dataset, batch_size=2, shuffle=True)


@pytest.fixture()
def val_dataloader() -> DataLoader:
    dataset = _TestDataset(size=4)
    return DataLoader(dataset, batch_size=2, shuffle=True)


def test_init(
    train_dataloader: DataLoader, val_dataloader: DataLoader
) -> None:
    mdl = torch.nn.Linear(2, 1)
    cfg = PyTorchTrainerConfig(
        net=mdl,
        train_data=train_dataloader,
        val_data=val_dataloader,
        a=1,
        b=2,
        c="3",
    )

    assert cfg["a"] == 1
    assert cfg["b"] == 2
    assert cfg["c"] == "3"
