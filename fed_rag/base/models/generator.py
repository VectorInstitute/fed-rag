"""Base Generator Model"""

from abc import abstractmethod
from typing import List

import numpy as np
import torch.nn as nn
from pydantic import BaseModel


class BaseGeneratorModel(BaseModel):
    net: nn.Module

    @abstractmethod
    def get_weights(self) -> List[np.ndarray]:
        """Get weights of the GeneratorModel's `net`.

        Returns:
            List[np.array]: List of weight parameters as `numpy.ndarray`s.
        """
