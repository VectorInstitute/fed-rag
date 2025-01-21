"""Base Generator Model"""

from abc import abstractmethod
from collections import OrderedDict
from typing import List

import numpy as np
import torch
import torch.nn as nn
from pydantic import BaseModel


class BaseGeneratorModel(BaseModel):
    net: nn.Module

    @abstractmethod
    def get_weights(self) -> List[np.ndarray]:
        """Get weights of the GeneratorModel's `net`.

        Implemented by the user.

        Returns:
            List[np.array]: List of weight parameters as `numpy.ndarray`s.
        """

    def set_weights(self, params: List[np.ndarray]) -> None:
        """Set the GeneratorModel's weights.

        Implemented by the user.

        Args:
            params (List[np.ndarray]): The list of weights that should be used to
            set the weights.
        """
        params_dict = zip(self.net.state_dict().keys(), params)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.net.load_state_dict(state_dict, strict=True)
