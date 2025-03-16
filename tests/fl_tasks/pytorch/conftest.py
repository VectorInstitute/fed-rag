"""PyTorchFLTask Unit Tests"""

from typing import Any, Callable

import numpy as np
import pytest
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from fed_rag.decorators import federate
from fed_rag.types import TestResult, TrainResult


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


@pytest.fixture()
def trainer() -> Callable:
    @federate.trainer.pytorch
    def fn(
        net: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
    ) -> TrainResult:
        return TrainResult(loss=0.0)

    return fn  # type: ignore


@pytest.fixture()
def tester() -> Callable:
    @federate.tester.pytorch
    def fn(
        net: nn.Module,
        test_loader: DataLoader,
    ) -> TestResult:
        return TestResult(loss=0.0, metrics={})

    return fn  # type: ignore


@pytest.fixture()
def mismatch_tester() -> Callable:
    @federate.tester.pytorch
    def fn(
        mdl: nn.Module,  # mismatch in name here
        test_loader: DataLoader,
    ) -> TestResult:
        return TestResult(loss=0.0, metrics={})

    return fn  # type: ignore
