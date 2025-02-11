"""PyTorch FL Task"""

import warnings
from typing import Any, Callable

import torch.nn as nn
from flwr.client import NumPyClient
from flwr.client.client import Client
from flwr.server.server import Server
from flwr.server.server_config import ServerConfig
from flwr.server.strategy import Strategy
from pydantic import BaseModel, ConfigDict, PrivateAttr
from torch.utils.data import DataLoader
from typing_extensions import Self

from fed_rag.base.fl_task import BaseFLTask, BaseFLTaskConfig
from fed_rag.exceptions import (
    MissingRequiredNetParam,
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


class BaseFLTaskBundle(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    net: nn.Module
    trainloader: DataLoader
    valloader: DataLoader
    trainer: Callable
    tester: Callable
    extra_test_kwargs: Any
    extra_train_kwargs: Any


class PyTorchFlowerClient(NumPyClient):
    def __init__(
        self,
        task_bundle: BaseFLTaskBundle,
    ) -> None:
        super().__init__()
        self.task_bundle = task_bundle

    def __getattr__(self, name: str) -> Any:
        if name in self.task_bundle.model_fields:
            return getattr(self.task_bundle, name)
        else:
            return super().__getattr__(name)

    # def get_weights(self) -> NDArrays:
    #     return [val.cpu().numpy() for _, val in self.net.state_dict().items()]

    # def set_weights(self, parameters: NDArrays) -> None:
    #     params_dict = zip(self.net.state_dict().keys(), parameters)
    #     state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    #     self.net.load_state_dict(state_dict, strict=True)

    # def fit(
    #     self, parameters: NDArrays, config: dict[str, Scalar]
    # ) -> tuple[NDArrays, int, dict[str, Scalar]]:
    #     self.set_weights(parameters)

    #     result: TrainResult = self.trainer(
    #         self.net,
    #         self.trainloader,
    #         self.valloader,
    #         **self.task_bundle.extra_train_kwargs,
    #     )
    #     return (
    #         self.get_weights(),
    #         len(self.trainloader.dataset),
    #         result.loss,
    #     )

    # def evaluate(
    #     self, parameters: NDArrays, config: dict[str, Scalar]
    # ) -> tuple[float, int, dict[str, Scalar]]:
    #     self.set_weights(parameters)
    #     result: TestResult = self.tester(self.net, self.valloader, self.device)
    #     return result.loss, len(self.valloader.dataset), result.metrics


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

    def server(
        self, strategy: Strategy, config: ServerConfig, **kwargs: Any
    ) -> Server | None:
        # validate kwargs
        if self._trainer_spec.net_parameter not in kwargs:
            msg = f"Please pass in a model using the model param name {self._trainer_spec.net_parameter}."
            raise MissingRequiredNetParam(msg)
        return None

    def client(self, **kwargs: Any) -> Client | None:
        # validate kwargs
        if self._trainer_spec.net_parameter not in kwargs:
            msg = f"Please pass in a model using the model param name {self._trainer_spec.net_parameter}."
            raise MissingRequiredNetParam(msg)
        # build bundle
        net = kwargs.pop(self._trainer_spec.net_parameter)
        trainloader = kwargs.pop(self._trainer_spec.train_data_param)
        valloader = kwargs.pop(self._trainer_spec.val_data_param)

        bundle = BaseFLTaskBundle(
            net=net,
            trainloader=trainloader,
            valloader=valloader,
            extra_train_kwargs=kwargs,
            extra_test_kwargs={},  # TODO make this functional or get rid of it
        )
        return PyTorchFlowerClient(task_bundle=bundle)

    def simulate(self, num_clients: int, **kwargs: Any) -> Any:
        raise NotImplementedError
