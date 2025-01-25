"""Base FL Task"""

from abc import abstractmethod
from typing import Any, Dict, Iterable, List, TypeAlias, Union

import numpy as np
from pydantic import BaseModel
from torch.utils.data import DataLoader

DataT: TypeAlias = Union[Iterable[Dict[str, Any]], DataLoader]


class BaseFLTask(BaseModel):
    model: Any
    training_loop: Any
