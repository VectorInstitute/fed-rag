"""PyTorch FL Task"""

from typing import Any, Callable

import torch.nn as nn
from flwr.client import NumPyClient
from pydantic import PrivateAttr

from fed_rag.base.fl_task import BaseFLTask, BaseFLTaskConfig
from fed_rag.inspectors.pytorch import (
    TesterSignatureSpec,
    TrainerSignatureSpec,
)


class PyTorchFLTaskConfig(BaseFLTaskConfig):
    pass


class PyTorchFLTask(BaseFLTask):
    _net: nn.Module = PrivateAttr()
    _client: NumPyClient = PrivateAttr()
    _trainer: Callable = PrivateAttr()
    _train_spec: TrainerSignatureSpec = PrivateAttr()
    _tester: Callable = PrivateAttr()
    _test_spec: TesterSignatureSpec = PrivateAttr()

    def __init__(
        self,
        net: nn.Module,
        trainer: Callable,
        train_spec: TrainerSignatureSpec,
        tester: Callable,
        test_spec: TesterSignatureSpec,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self._net = net
        self._trainer = trainer
        self._train_spec = train_spec
        self._tester = tester
        self._test_spec = test_spec

        # build flwr NumpyClient
        self._client = ...

    @property
    def net(self) -> nn.Module:
        return self.net

    @property
    def training_loop(self) -> Callable:
        return self._trainer

    @classmethod
    def from_trainer_and_tester(
        cls, trainer: Callable, tester: Callable
    ) -> Any:
        return super().from_trainer_and_tester(trainer, tester)

    @classmethod
    def from_configs(cls, trainer_cfg: Any, tester_cfg: Any) -> Any:
        return super().from_configs(trainer_cfg, tester_cfg)
