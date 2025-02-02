"""PyTorch FL Task"""

from typing import Any, Callable

import torch.nn as nn
from flwr.client import NumPyClient
from pydantic import PrivateAttr
from typing_extensions import Self

from fed_rag.base.fl_task import BaseFLTask, BaseFLTaskConfig
from fed_rag.exceptions import MissingTesterSpec, MissingTrainerSpec
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
    _trainer_spec: TrainerSignatureSpec = PrivateAttr()
    _tester: Callable = PrivateAttr()
    _tester_spec: TesterSignatureSpec = PrivateAttr()

    def __init__(
        self,
        net: nn.Module,
        trainer: Callable,
        trainer_spec: TrainerSignatureSpec,
        tester: Callable,
        tester_spec: TesterSignatureSpec,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self._net = net
        self._trainer = trainer
        self._trainer_spec = trainer_spec
        self._tester = tester
        self._tester_spec = tester_spec

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
        cls, net: nn.Module, trainer: Callable, tester: Callable
    ) -> Self:
        # extract trainer spec
        try:
            trainer_spec = getattr(trainer, "__fl_task_trainer_config")
        except AttributeError:
            msg = "Cannot extract `TrainerSignatureSpec` from supplied `trainer`."
            raise MissingTrainerSpec(msg)

        # extract tester spec
        try:
            tester_spec = getattr(trainer, "__fl_task_tester_config")
        except AttributeError:
            msg = (
                "Cannot extract `TesterSignatureSpec` from supplied `tester`."
            )
            raise MissingTesterSpec(msg)

        return cls(
            net=net,
            trainer=trainer,
            trainer_spec=trainer_spec,
            tester=tester,
            tester_spec=tester_spec,
        )

    @classmethod
    def from_configs(cls, trainer_cfg: Any, tester_cfg: Any) -> Any:
        return super().from_configs(trainer_cfg, tester_cfg)
