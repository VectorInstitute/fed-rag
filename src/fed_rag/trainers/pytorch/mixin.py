"""PyTorch Trainer Mixin"""

from abc import ABC
from typing import Any, Protocol, runtime_checkable

import torch.nn as nn
from pydantic import BaseModel, ConfigDict
from torch.utils.data import DataLoader, Dataset

from fed_rag.exceptions import InconsistentDatasetError

from .training_args import TrainingArgs


# Define the protocol for runtime checking
@runtime_checkable
class PyTorchTrainerProtocol(Protocol):
    train_dataset: Dataset
    training_arguments: TrainingArgs | None
    train_dataloader: DataLoader

    def model(self) -> nn.Module:
        pass  # pragma: no cover


class PyTorchTrainerMixin(BaseModel, ABC):
    """HuggingFace Trainer Mixin."""

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
    )
    train_dataset: Dataset
    train_dataloader: DataLoader
    training_arguments: TrainingArgs | None = None

    def __init__(
        self,
        train_dataloader: DataLoader,
        train_dataset: Dataset | None = None,
        training_arguments: TrainingArgs | None = None,
        **kwargs: Any,
    ):
        if train_dataset is None:
            train_dataset = train_dataloader.dataset
        else:
            # ensure consistency between loader.dataset and the supplied one
            if id(train_dataset) != id(train_dataloader.dataset):
                raise InconsistentDatasetError(
                    "Inconsistent datasets detected between supplied `train_dataset` and that "
                    "associated with the `train_dataloader`. These two datasets must be the same."
                )

        super().__init__(
            train_dataset=train_dataset,
            train_dataloader=train_dataloader,
            training_arguments=training_arguments,
            **kwargs,
        )
