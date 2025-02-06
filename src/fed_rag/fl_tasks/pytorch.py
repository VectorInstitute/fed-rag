"""PyTorch FL Task"""

import warnings
from typing import Any, Callable, OrderedDict

import torch
import torch.nn as nn
from flwr.client import NumPyClient
from flwr.common.typing import NDArrays, Scalar
from pydantic import PrivateAttr
from typing_extensions import Self

from fed_rag.base.fl_task import BaseFLTask, BaseFLTaskConfig
from fed_rag.exceptions import (
    MissingTesterSpec,
    MissingTrainerSpec,
    UnequalNetParamWarning,
)
from fed_rag.inspectors.pytorch import (
    TesterSignatureSpec,
    TrainerSignatureSpec,
)


class PyTorchFLTaskConfig(BaseFLTaskConfig):
    pass


class PyTorchFlowerClient(NumPyClient):
    def __init__(
        self,
        net: nn.Module,
        trainer: Callable,
        trainer_spec: TrainerSignatureSpec,
        tester: Callable,
        tester_spec: TesterSignatureSpec,
    ) -> None:
        super().__init__()
        self.net = net
        self.trainer = trainer
        self.trainer_spec = trainer_spec
        self.tester = tester
        self.tester_spec = tester_spec

    def get_weights(self) -> NDArrays:
        return [val.cpu().numpy() for _, val in self.net.state_dict().items()]

    def set_weights(self, parameters: NDArrays) -> None:
        params_dict = zip(self.net.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.net.load_state_dict(state_dict, strict=True)

    def fit(
        self, parameters: NDArrays, config: dict[str, Scalar]
    ) -> tuple[NDArrays, int, dict[str, Scalar]]:
        self.set_weights(parameters)

        results = self.trainer(
            self.net,
            self.trainloader,
            self.valloader,
            self.local_epochs,
            self.device,
        )
        return (
            self.get_weights(),
            len(self.trainloader.dataset),
            results,
        )

    def evaluate(
        self, parameters: NDArrays, config: dict[str, Scalar]
    ) -> tuple[float, int, dict[str, Scalar]]:
        self.set_weights(parameters)
        loss, accuracy = self.tester(self.net, self.valloader, self.device)
        return loss, len(self.valloader.dataset), {"accuracy": accuracy}


def _build_client(
    net: nn.Module,
    trainer: Callable,
    trainer_spec: TrainerSignatureSpec,
    tester: Callable,
    tester_spec: TesterSignatureSpec,
) -> NumPyClient:
    ...


class PyTorchFLTask(BaseFLTask):
    _client: NumPyClient = PrivateAttr()
    _trainer: Callable = PrivateAttr()
    _trainer_spec: TrainerSignatureSpec = PrivateAttr()
    _tester: Callable = PrivateAttr()
    _tester_spec: TesterSignatureSpec = PrivateAttr()

    def __init__(
        self,
        trainer: Callable,
        trainer_spec: TrainerSignatureSpec,
        tester: Callable,
        tester_spec: TesterSignatureSpec,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self._trainer = trainer
        self._trainer_spec = trainer_spec
        self._tester = tester
        self._tester_spec = tester_spec

    @property
    def training_loop(self) -> Callable:
        return self._trainer

    @classmethod
    def from_trainer_and_tester(
        cls, trainer: Callable, tester: Callable
    ) -> Self:
        # extract trainer spec
        try:
            trainer_spec: TrainerSignatureSpec = getattr(
                trainer, "__fl_task_trainer_config"
            )
        except AttributeError:
            msg = "Cannot extract `TrainerSignatureSpec` from supplied `trainer`."
            raise MissingTrainerSpec(msg)

        # extract tester spec
        try:
            tester_spec: TesterSignatureSpec = getattr(
                tester, "__fl_task_tester_config"
            )
        except AttributeError:
            msg = (
                "Cannot extract `TesterSignatureSpec` from supplied `tester`."
            )
            raise MissingTesterSpec(msg)

        if trainer_spec.net_parameter != tester_spec.net_parameter:
            msg = (
                "`trainer`'s model parameter name is not the same as that for `tester`. "
                "Will use the name supplied in `trainer`."
            )
            warnings.warn(msg, UnequalNetParamWarning)

        return cls(
            trainer=trainer,
            trainer_spec=trainer_spec,
            tester=tester,
            tester_spec=tester_spec,
        )

    @classmethod
    def from_configs(cls, trainer_cfg: Any, tester_cfg: Any) -> Any:
        return super().from_configs(trainer_cfg, tester_cfg)
