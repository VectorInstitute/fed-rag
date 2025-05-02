from typing import Any

import numpy as np
import pytest
from torch.utils.data import DataLoader, Dataset

from fed_rag.base.trainer import BaseGeneratorTrainer, BaseRetrieverTrainer
from fed_rag.trainers.pytorch.mixin import PyTorchTrainerMixin
from fed_rag.types.results import TestResult, TrainResult


class TestRetrieverTrainer(PyTorchTrainerMixin, BaseRetrieverTrainer):
    __test__ = (
        False  # needed for Pytest collision. Avoids PytestCollectionWarning
    )

    def train(self) -> TrainResult:
        return TrainResult(loss=0.42)

    def evaluate(self) -> TestResult:
        return TestResult(loss=0.42)


class TestGeneratorTrainer(PyTorchTrainerMixin, BaseGeneratorTrainer):
    __test__ = (
        False  # needed for Pytest collision. Avoids PytestCollectionWarning
    )

    def train(self) -> TrainResult:
        return TrainResult(loss=0.42)

    def evaluate(self) -> TestResult:
        return TestResult(loss=0.42)


class _TestDataset(Dataset):
    def __init__(self, size: int) -> None:
        self.features = np.random.rand(size, 2)
        self.labels = np.random.choice(2, size=size)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> tuple[np.ndarray, Any]:
        return self.features[index], self.labels[index]


@pytest.fixture()
def train_dataset() -> Dataset:
    return _TestDataset(size=10)


@pytest.fixture()
def another_train_dataset() -> Dataset:
    return _TestDataset(size=10)


@pytest.fixture()
def train_dataloader(train_dataset: Dataset) -> DataLoader:
    return DataLoader(train_dataset, batch_size=2, shuffle=True)
