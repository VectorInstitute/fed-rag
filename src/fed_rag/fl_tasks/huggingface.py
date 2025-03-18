"""HuggingFace FL Task.

NOTE: Using this module requires the `huggingface` extra to be installed.
"""

import warnings
from typing import Any, Callable, Dict, OrderedDict, TypeAlias, cast

import torch
from datasets import Dataset
from flwr.client import NumPyClient
from flwr.client.client import Client
from flwr.common import NDArrays, Scalar
from flwr.common.parameter import ndarrays_to_parameters
from flwr.server.client_manager import ClientManager, SimpleClientManager
from flwr.server.server import Server
from flwr.server.strategy import FedAvg, Strategy
from pydantic import BaseModel, ConfigDict, PrivateAttr

# check if huggingface extra was installed
try:
    from peft import PeftModel
    from peft.utils import get_peft_model_state_dict, set_peft_model_state_dict
    from sentence_transformers import SentenceTransformer
    from transformers import PreTrainedModel
except ModuleNotFoundError:
    msg = (
        "This decorator requires `huggingface` extra to be installed. "
        "To fix please run `pip install fed-rag[huggingface]`."
    )
    raise ValueError(msg)


from typing_extensions import Self

from fed_rag.base.fl_task import BaseFLTask
from fed_rag.exceptions import (
    MissingRequiredNetParam,
    MissingTesterSpec,
    MissingTrainerSpec,
    NetTypeMismatch,
    UnequalNetParamWarning,
)
from fed_rag.inspectors.pytorch import (
    TesterSignatureSpec,
    TrainerSignatureSpec,
)
from fed_rag.types import TestResult, TrainResult


class BaseFLTaskBundle(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    net: SentenceTransformer | PreTrainedModel | PeftModel
    train_dataset: Dataset
    val_dataset: Dataset
    trainer: Callable
    tester: Callable
    extra_test_kwargs: Any
    extra_train_kwargs: Any


def _get_weights(
    net: SentenceTransformer | PreTrainedModel | PeftModel,
) -> NDArrays:
    """Get weights from net.

    NOTE: SentenceTransformer and PreTrainedModel are subclasses of torch.nn.Module
    """
    if isinstance(net, PeftModel):
        state_dict = get_peft_model_state_dict(net)
        state_dict = cast(Dict[str, Any], state_dict)
    elif isinstance(net, SentenceTransformer) or isinstance(
        net, PreTrainedModel
    ):
        state_dict = net.state_dict()
    else:
        raise ValueError("Invalid `net` type.")
    return [val.cpu().numpy() for _, val in state_dict.items()]


class HuggingFaceFlowerClient(NumPyClient):
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

    def get_weights(self) -> NDArrays:
        return _get_weights(self.net)

    def set_weights(self, parameters: NDArrays) -> None:
        if isinstance(self.net, PeftModel):
            # get state dict
            state_dict = get_peft_model_state_dict(self.net)
            state_dict = cast(Dict[str, Any], state_dict)

            # update state dict with supplied parameters
            params_dict = zip(state_dict.keys(), parameters)
            state_dict = OrderedDict(
                {k: torch.tensor(v) for k, v in params_dict}
            )
            set_peft_model_state_dict(self.net, state_dict)
        else:  # SentenceTransformer | PreTrainedModel
            # get state dict
            state_dict = self.net.state_dict()

            # update state dict with supplied parameters
            params_dict = zip(state_dict.keys(), parameters)
            state_dict = OrderedDict(
                {k: torch.tensor(v) for k, v in params_dict}
            )
            self.net.load_state_dict(state_dict, strict=True)

    def fit(
        self, parameters: NDArrays, config: dict[str, Scalar]
    ) -> tuple[NDArrays, int, dict[str, Scalar]]:
        self.set_weights(parameters)

        result: TrainResult = self.trainer(
            self.net,
            self.train_dataset,
            self.val_dataset,
            **self.task_bundle.extra_train_kwargs,
        )
        return (
            self.get_weights(),
            len(self.train_dataset),
            {"loss": result.loss},
        )

    def evaluate(
        self, parameters: NDArrays, config: dict[str, Scalar]
    ) -> tuple[float, int, dict[str, Scalar]]:
        self.set_weights(parameters)
        result: TestResult = self.tester(
            self.net, self.val_dataset, **self.extra_test_kwargs
        )
        return result.loss, len(self.val_dataset), result.metrics


HuggingFaceFlowerServer: TypeAlias = Server


class HuggingFaceFLTask(BaseFLTask):
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
        if (
            trainer_spec.net_parameter_class_name
            != tester_spec.net_parameter_class_name
        ):
            msg = (
                "`trainer`'s model class is not the same as that for `tester`."
            )
            raise NetTypeMismatch(msg)

        if trainer_spec.net_parameter != tester_spec.net_parameter:
            msg = (
                "`trainer`'s model parameter name is not the same as that for `tester`. "
                "Will use the name supplied in `trainer`."
            )
            warnings.warn(msg, UnequalNetParamWarning)

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

        return cls(
            trainer=trainer,
            trainer_spec=trainer_spec,
            tester=tester,
            tester_spec=tester_spec,
        )

    def server(
        self,
        strategy: Strategy | None = None,
        client_manager: ClientManager | None = None,
        **kwargs: Any,
    ) -> HuggingFaceFlowerServer | None:
        if strategy is None:
            if self._trainer_spec.net_parameter not in kwargs:
                msg = f"Please pass in a model using the model param name {self._trainer_spec.net_parameter}."
                raise MissingRequiredNetParam(msg)

            model = kwargs.pop(self._trainer_spec.net_parameter)

            ndarrays = _get_weights(model)
            parameters = ndarrays_to_parameters(ndarrays)
            strategy = FedAvg(
                fraction_evaluate=1.0,
                initial_parameters=parameters,
            )

        if client_manager is None:
            client_manager = SimpleClientManager()

        return HuggingFaceFlowerServer(
            client_manager=client_manager, strategy=strategy
        )

    def client(self, **kwargs: Any) -> Client | None:
        # validate kwargs
        if self._trainer_spec.net_parameter not in kwargs:
            msg = f"Please pass in a model using the model param name {self._trainer_spec.net_parameter}."
            raise MissingRequiredNetParam(msg)
        # build bundle
        net = kwargs.pop(self._trainer_spec.net_parameter)
        train_dataset = kwargs.pop(self._trainer_spec.train_data_param)
        val_dataset = kwargs.pop(self._trainer_spec.val_data_param)

        bundle = BaseFLTaskBundle(
            net=net,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            extra_train_kwargs=kwargs,
            extra_test_kwargs={},  # TODO make this functional or get rid of it
            trainer=self._trainer,
            tester=self._tester,
        )
        return HuggingFaceFlowerClient(task_bundle=bundle)

    def simulate(self, num_clients: int, **kwargs: Any) -> Any:
        raise NotImplementedError

    @classmethod
    def from_configs(cls, trainer_cfg: Any, tester_cfg: Any) -> Any:
        raise NotImplementedError
