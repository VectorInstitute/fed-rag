"""PyTorch FL Task"""

import torch.nn as nn

from fed_rag.base.fl_task import BaseFLTask, BaseFLTaskConfig


class PyTorchFLTaskConfig(BaseFLTaskConfig):
    pass


class PyTorchFLTask(BaseFLTask):
    net: nn.Module
