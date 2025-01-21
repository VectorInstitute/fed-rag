"""Base Generator Model"""

from abc import abstractmethod
from typing import Any, Dict, Iterable, List, TypeAlias, Union

import numpy as np
from pydantic import BaseModel
from torch.utils.data import DataLoader

DataT: TypeAlias = Union[Iterable[Dict[str, Any]], DataLoader]


class BaseTaskModel(BaseModel):
    @abstractmethod
    def get_weights(self) -> List[np.ndarray]:
        """Get weights of the TorchModel's `net`.

        Implemented by the user.

        Returns:
            List[np.array]: List of weight parameters as `numpy.ndarray`s.
        """

    @abstractmethod
    def set_weights(self, params: List[np.ndarray]) -> None:
        """Set the TorchModel's weights.

        Implemented by the user.

        Args:
            params (List[np.ndarray]): The list of weights that should be used to
            set the weights.
        """

    @abstractmethod
    def train(
        self, train_data: DataT, val_data: DataT, **kwargs: Any
    ) -> dict[str, float]:
        """Train the base model using the supplied `trainer_callback`.

        Args:
            train_loader (DataT): The data loader for training examples.
            val_loader (DataT): The data loader for validation examples.

        Returns:
            dict[str, float]: A dictionary containing the results of the training.
        """
