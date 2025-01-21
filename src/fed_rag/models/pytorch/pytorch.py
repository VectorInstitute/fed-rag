"""PyTorch Task Model"""

from typing import Any, List, OrderedDict

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from fed_rag.base.models.base_task_model import BaseTaskModel


class PyTorchTaskModel(BaseTaskModel):
    net: nn.Module

    def get_weights(self) -> List[np.ndarray]:
        """Get weights of the TorchModel's `net`.

        Implemented by the user.

        Returns:
            List[np.array]: List of weight parameters as `numpy.ndarray`s.
        """
        return [val.cpu().numpy() for _, val in self.net.state_dict().items()]

    def set_weights(self, params: List[np.ndarray]) -> None:
        """Set the TorchModel's weights.

        Implemented by the user.

        Args:
            params (List[np.ndarray]): The list of weights that should be used to
            set the weights.
        """
        params_dict = zip(self.net.state_dict().keys(), params)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.net.load_state_dict(state_dict, strict=True)

    def train(
        self, train_data: DataLoader, val_data: DataLoader, kwargs: Any
    ) -> dict[str, float]:
        """Train the base model using the supplied `trainer_callback`.

        Args:
            train_loader (DataT): The data loader for training examples.
            val_loader (DataT): The data loader for validation examples.
            kwargs (Any): Any kwargs to pass to training and validiation processes.

        Returns:
            dict[str, float]: A dictionary containing the results of the training.
        """
        return self.train_callback(train_data=train_data, val_data=val_data)
