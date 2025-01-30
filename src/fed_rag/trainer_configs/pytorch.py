"""PyTorch Trainer Config"""

import torch
from torch.utils.data import DataLoader

from fed_rag.base.trainer_config import BaseTrainerConfig


class PyTorchTrainerConfig(BaseTrainerConfig):
    net: torch.nn.Module
    train_data: DataLoader
    val_data: DataLoader
