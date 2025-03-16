"""Decorators unit tests for HuggingFace"""

from typing import Any

import pytest
from datasets import Dataset
from peft import PeftModel
from sentence_transformers import SentenceTransformer
from transformers import PreTrainedModel

from fed_rag.decorators import federate
from fed_rag.exceptions.inspectors import (
    InvalidReturnType,
    MissingDataParam,
    MissingMultipleDataParams,
    MissingNetParam,
)
from fed_rag.inspectors.huggingface import (
    TesterSignatureSpec,
    TrainerSignatureSpec,
)
from fed_rag.types import TestResult, TrainResult


## For All
def test_decorated_trainer_raises_missing_net_param_error() -> None:
    def train_loop(
        train_dataset: Dataset,
        val_dataset: Dataset,
    ) -> TrainResult:
        pass

    with pytest.raises(MissingNetParam):
        federate.trainer.huggingface(train_loop)


## PretrainedModel
### Trainer
def test_decorated_trainer_hf_pretrained_model() -> None:
    def train_loop(
        net: PreTrainedModel,
        train_dataset: Dataset,
        val_dataset: Dataset,
        extra_param_1: int,
        extra_param_2: float | None,
    ) -> TrainResult:
        pass

    decorated = federate.trainer.huggingface(train_loop)
    config: TrainerSignatureSpec = getattr(
        decorated, "__fl_task_trainer_config"
    )
    assert config.net_parameter == "net"
    assert config.train_data_param == "train_dataset"
    assert config.val_data_param == "val_dataset"
    assert config.extra_train_kwargs == ["extra_param_1", "extra_param_2"]
    assert config.net_parameter_class_name == "PreTrainedModel"


def test_decorated_trainer_raises_invalid_return_type_error_hf_pretrained() -> (
    None
):
    def train_loop(
        net: PreTrainedModel,
        train_dataset: Dataset,
        val_dataset: Dataset,
        extra_param_1: int,
        extra_param_2: float | None,
    ) -> Any:
        pass

    with pytest.raises(
        InvalidReturnType,
        match="Trainer should return a fed_rag.types.TrainResult or a subclass of it.",
    ):
        federate.trainer.huggingface(train_loop)


def test_decorated_trainer_fails_to_find_two_data_params_hf_pretrained() -> (
    None
):
    def train_loop(
        model: PreTrainedModel,
    ) -> TrainResult:
        pass

    msg = (
        "Inspection failed to find two data params for train and val datasets."
        "For HuggingFace these params must be of type `datasets.Dataset`"
    )
    with pytest.raises(MissingMultipleDataParams, match=msg):
        federate.trainer.huggingface(train_loop)


def test_decorated_trainer_fails_due_to_missing_data_loader_hf_pretrained() -> (
    None
):
    def train_loop(model: PreTrainedModel, val_loader: Dataset) -> TrainResult:
        pass

    msg = (
        "Inspection found one data param but failed to find another. "
        "Two data params are required for train and val datasets."
        "For HuggingFace these params must be of type `datasets.Dataset`"
    )
    with pytest.raises(MissingDataParam, match=msg):
        federate.trainer.huggingface(train_loop)


def test_decorated_trainer_from_instance_method_hf_pretrained() -> None:
    class _TestClass:
        @federate.trainer.huggingface
        def train_loop(
            self,
            net: PreTrainedModel,
            train_loader: Dataset,
            val_loader: Dataset,
            extra_param_1: int,
            extra_param_2: float | None,
        ) -> TrainResult:
            pass

    obj = _TestClass()
    config: TrainerSignatureSpec = getattr(
        obj.train_loop, "__fl_task_trainer_config"
    )
    assert config.net_parameter == "net"
    assert config.train_data_param == "train_loader"
    assert config.val_data_param == "val_loader"
    assert config.extra_train_kwargs == ["extra_param_1", "extra_param_2"]
    assert config.net_parameter_class_name == "PreTrainedModel"


def test_decorated_trainer_from_class_method_hf_pretrained() -> None:
    class _TestClass:
        @classmethod
        @federate.trainer.huggingface
        def train_loop(
            cls,
            net: PreTrainedModel,
            train_loader: Dataset,
            val_loader: Dataset,
            extra_param_1: int,
            extra_param_2: float | None,
        ) -> TrainResult:
            pass

    obj = _TestClass()
    config: TrainerSignatureSpec = getattr(
        obj.train_loop, "__fl_task_trainer_config"
    )
    assert config.net_parameter == "net"
    assert config.train_data_param == "train_loader"
    assert config.val_data_param == "val_loader"
    assert config.extra_train_kwargs == ["extra_param_1", "extra_param_2"]
    assert config.net_parameter_class_name == "PreTrainedModel"


### Tester
def test_decorated_tester_hf_pretrained() -> None:
    def fn(
        mdl: PreTrainedModel,
        test_loader: Dataset,
        extra_param_1: int,
        extra_param_2: float | None,
    ) -> TestResult:
        pass

    decorated = federate.tester.huggingface(fn)
    config: TesterSignatureSpec = getattr(decorated, "__fl_task_tester_config")
    assert config.net_parameter == "mdl"
    assert config.test_data_param == "test_loader"
    assert config.extra_test_kwargs == ["extra_param_1", "extra_param_2"]
    assert config.net_parameter_class_name == "PreTrainedModel"


def test_decorated_tester_raises_invalid_return_type_hf_pretrained() -> None:
    def fn(
        mdl: PreTrainedModel,
        test_loader: Dataset,
        extra_param_1: int,
        extra_param_2: float | None,
    ) -> Any:
        pass

    with pytest.raises(
        InvalidReturnType,
        match="Tester should return a fed_rag.types.TestResult or a subclass of it.",
    ):
        federate.tester.huggingface(fn)


def test_decorated_tester_raises_missing_net_param_error_hf_pretrained() -> (
    None
):
    def fn(
        test_loader: Dataset,
    ) -> TestResult:
        pass

    with pytest.raises(MissingNetParam):
        federate.tester.huggingface(fn)


def test_decorated_tester_fails_to_find_a_data_params_hf_pretrained() -> None:
    def fn(
        model: PreTrainedModel,
    ) -> TestResult:
        pass

    msg = (
        "Inspection failed to find a data param for a test dataset."
        "For HuggingFace these params must be of type `datasets.Dataset`"
    )
    with pytest.raises(MissingDataParam, match=msg):
        federate.tester.huggingface(fn)


def test_decorated_tester_from_instance_method_hf_pretrained() -> None:
    class _TestClass:
        @federate.tester.huggingface
        def fn(
            self,
            mdl: PreTrainedModel,
            test_loader: Dataset,
            extra_param_1: int,
        ) -> TestResult:
            pass

    obj = _TestClass()
    config: TesterSignatureSpec = getattr(obj.fn, "__fl_task_tester_config")
    assert config.net_parameter == "mdl"
    assert config.test_data_param == "test_loader"
    assert config.extra_test_kwargs == ["extra_param_1"]
    assert config.net_parameter_class_name == "PreTrainedModel"


def test_decorated_tester_from_class_method_hf_pretrained() -> None:
    class _TestClass:
        @classmethod
        @federate.tester.huggingface
        def fn(
            cls,
            mdl: PreTrainedModel,
            test_loader: Dataset,
            extra_param_1: int,
        ) -> TestResult:
            pass

    config: TesterSignatureSpec = getattr(
        _TestClass.fn, "__fl_task_tester_config"
    )
    assert config.net_parameter == "mdl"
    assert config.test_data_param == "test_loader"
    assert config.extra_test_kwargs == ["extra_param_1"]
    assert config.net_parameter_class_name == "PreTrainedModel"


## Sentence Transformers
### Trainer
def test_decorated_trainer_hf_st() -> None:
    def train_loop(
        net: SentenceTransformer,
        train_dataset: Dataset,
        val_dataset: Dataset,
        extra_param_1: int,
        extra_param_2: float | None,
    ) -> TrainResult:
        pass

    decorated = federate.trainer.huggingface(train_loop)
    config: TrainerSignatureSpec = getattr(
        decorated, "__fl_task_trainer_config"
    )
    assert config.net_parameter == "net"
    assert config.train_data_param == "train_dataset"
    assert config.val_data_param == "val_dataset"
    assert config.extra_train_kwargs == ["extra_param_1", "extra_param_2"]
    assert config.net_parameter_class_name == "SentenceTransformer"


def test_decorated_trainer_raises_invalid_return_type_error_hf_st() -> None:
    def train_loop(
        net: SentenceTransformer,
        train_dataset: Dataset,
        val_dataset: Dataset,
        extra_param_1: int,
        extra_param_2: float | None,
    ) -> Any:
        pass

    with pytest.raises(
        InvalidReturnType,
        match="Trainer should return a fed_rag.types.TrainResult or a subclass of it.",
    ):
        federate.trainer.huggingface(train_loop)


def test_decorated_trainer_fails_to_find_two_data_params_hf_st() -> None:
    def train_loop(
        model: SentenceTransformer,
    ) -> TrainResult:
        pass

    msg = (
        "Inspection failed to find two data params for train and val datasets."
        "For HuggingFace these params must be of type `datasets.Dataset`"
    )
    with pytest.raises(MissingMultipleDataParams, match=msg):
        federate.trainer.huggingface(train_loop)


def test_decorated_trainer_fails_due_to_missing_data_loader_hf_st() -> None:
    def train_loop(
        model: SentenceTransformer, val_loader: Dataset
    ) -> TrainResult:
        pass

    msg = (
        "Inspection found one data param but failed to find another. "
        "Two data params are required for train and val datasets."
        "For HuggingFace these params must be of type `datasets.Dataset`"
    )
    with pytest.raises(MissingDataParam, match=msg):
        federate.trainer.huggingface(train_loop)


def test_decorated_trainer_from_instance_method_hf_st() -> None:
    class _TestClass:
        @federate.trainer.huggingface
        def train_loop(
            self,
            net: SentenceTransformer,
            train_loader: Dataset,
            val_loader: Dataset,
            extra_param_1: int,
            extra_param_2: float | None,
        ) -> TrainResult:
            pass

    obj = _TestClass()
    config: TrainerSignatureSpec = getattr(
        obj.train_loop, "__fl_task_trainer_config"
    )
    assert config.net_parameter == "net"
    assert config.train_data_param == "train_loader"
    assert config.val_data_param == "val_loader"
    assert config.extra_train_kwargs == ["extra_param_1", "extra_param_2"]
    assert config.net_parameter_class_name == "SentenceTransformer"


def test_decorated_trainer_from_class_method_hf_st() -> None:
    class _TestClass:
        @classmethod
        @federate.trainer.huggingface
        def train_loop(
            cls,
            net: SentenceTransformer,
            train_loader: Dataset,
            val_loader: Dataset,
            extra_param_1: int,
            extra_param_2: float | None,
        ) -> TrainResult:
            pass

    obj = _TestClass()
    config: TrainerSignatureSpec = getattr(
        obj.train_loop, "__fl_task_trainer_config"
    )
    assert config.net_parameter == "net"
    assert config.train_data_param == "train_loader"
    assert config.val_data_param == "val_loader"
    assert config.extra_train_kwargs == ["extra_param_1", "extra_param_2"]
    assert config.net_parameter_class_name == "SentenceTransformer"


### Tester
def test_decorated_tester_hf_st() -> None:
    def fn(
        mdl: SentenceTransformer,
        test_loader: Dataset,
        extra_param_1: int,
        extra_param_2: float | None,
    ) -> TestResult:
        pass

    decorated = federate.tester.huggingface(fn)
    config: TesterSignatureSpec = getattr(decorated, "__fl_task_tester_config")
    assert config.net_parameter == "mdl"
    assert config.test_data_param == "test_loader"
    assert config.extra_test_kwargs == ["extra_param_1", "extra_param_2"]
    assert config.net_parameter_class_name == "SentenceTransformer"


def test_decorated_tester_raises_invalid_return_type_hf_st() -> None:
    def fn(
        mdl: SentenceTransformer,
        test_loader: Dataset,
        extra_param_1: int,
        extra_param_2: float | None,
    ) -> Any:
        pass

    with pytest.raises(
        InvalidReturnType,
        match="Tester should return a fed_rag.types.TestResult or a subclass of it.",
    ):
        federate.tester.huggingface(fn)


def test_decorated_tester_fails_to_find_a_data_params_hf_st() -> None:
    def fn(
        model: SentenceTransformer,
    ) -> TestResult:
        pass

    msg = (
        "Inspection failed to find a data param for a test dataset."
        "For HuggingFace these params must be of type `datasets.Dataset`"
    )
    with pytest.raises(MissingDataParam, match=msg):
        federate.tester.huggingface(fn)


def test_decorated_tester_from_instance_method_hf_st() -> None:
    class _TestClass:
        @federate.tester.huggingface
        def fn(
            self,
            mdl: SentenceTransformer,
            test_loader: Dataset,
            extra_param_1: int,
        ) -> TestResult:
            pass

    obj = _TestClass()
    config: TesterSignatureSpec = getattr(obj.fn, "__fl_task_tester_config")
    assert config.net_parameter == "mdl"
    assert config.test_data_param == "test_loader"
    assert config.extra_test_kwargs == ["extra_param_1"]
    assert config.net_parameter_class_name == "SentenceTransformer"


def test_decorated_tester_from_class_method_hf_st() -> None:
    class _TestClass:
        @classmethod
        @federate.tester.huggingface
        def fn(
            cls,
            mdl: SentenceTransformer,
            test_loader: Dataset,
            extra_param_1: int,
        ) -> TestResult:
            pass

    config: TesterSignatureSpec = getattr(
        _TestClass.fn, "__fl_task_tester_config"
    )
    assert config.net_parameter == "mdl"
    assert config.test_data_param == "test_loader"
    assert config.extra_test_kwargs == ["extra_param_1"]
    assert config.net_parameter_class_name == "SentenceTransformer"


## PeftModel
### Trainer
def test_decorated_trainer_hf_peft() -> None:
    def train_loop(
        net: PeftModel,
        train_dataset: Dataset,
        val_dataset: Dataset,
        extra_param_1: int,
        extra_param_2: float | None,
    ) -> TrainResult:
        pass

    decorated = federate.trainer.huggingface(train_loop)
    config: TrainerSignatureSpec = getattr(
        decorated, "__fl_task_trainer_config"
    )
    assert config.net_parameter == "net"
    assert config.train_data_param == "train_dataset"
    assert config.val_data_param == "val_dataset"
    assert config.extra_train_kwargs == ["extra_param_1", "extra_param_2"]
    assert config.net_parameter_class_name == "PeftModel"


def test_decorated_trainer_raises_invalid_return_type_error_hf_peft() -> None:
    def train_loop(
        net: PeftModel,
        train_dataset: Dataset,
        val_dataset: Dataset,
        extra_param_1: int,
        extra_param_2: float | None,
    ) -> Any:
        pass

    with pytest.raises(
        InvalidReturnType,
        match="Trainer should return a fed_rag.types.TrainResult or a subclass of it.",
    ):
        federate.trainer.huggingface(train_loop)


def test_decorated_trainer_fails_to_find_two_data_params_hf_peft() -> None:
    def train_loop(
        model: PeftModel,
    ) -> TrainResult:
        pass

    msg = (
        "Inspection failed to find two data params for train and val datasets."
        "For HuggingFace these params must be of type `datasets.Dataset`"
    )
    with pytest.raises(MissingMultipleDataParams, match=msg):
        federate.trainer.huggingface(train_loop)


def test_decorated_trainer_fails_due_to_missing_data_loader_hf_peft() -> None:
    def train_loop(model: PeftModel, val_loader: Dataset) -> TrainResult:
        pass

    msg = (
        "Inspection found one data param but failed to find another. "
        "Two data params are required for train and val datasets."
        "For HuggingFace these params must be of type `datasets.Dataset`"
    )
    with pytest.raises(MissingDataParam, match=msg):
        federate.trainer.huggingface(train_loop)


def test_decorated_trainer_from_instance_method_hf_peft() -> None:
    class _TestClass:
        @federate.trainer.huggingface
        def train_loop(
            self,
            net: PeftModel,
            train_loader: Dataset,
            val_loader: Dataset,
            extra_param_1: int,
            extra_param_2: float | None,
        ) -> TrainResult:
            pass

    obj = _TestClass()
    config: TrainerSignatureSpec = getattr(
        obj.train_loop, "__fl_task_trainer_config"
    )
    assert config.net_parameter == "net"
    assert config.train_data_param == "train_loader"
    assert config.val_data_param == "val_loader"
    assert config.extra_train_kwargs == ["extra_param_1", "extra_param_2"]
    assert config.net_parameter_class_name == "PeftModel"


def test_decorated_trainer_from_class_method_hf_peft() -> None:
    class _TestClass:
        @classmethod
        @federate.trainer.huggingface
        def train_loop(
            cls,
            net: PeftModel,
            train_loader: Dataset,
            val_loader: Dataset,
            extra_param_1: int,
            extra_param_2: float | None,
        ) -> TrainResult:
            pass

    obj = _TestClass()
    config: TrainerSignatureSpec = getattr(
        obj.train_loop, "__fl_task_trainer_config"
    )
    assert config.net_parameter == "net"
    assert config.train_data_param == "train_loader"
    assert config.val_data_param == "val_loader"
    assert config.extra_train_kwargs == ["extra_param_1", "extra_param_2"]
    assert config.net_parameter_class_name == "PeftModel"


### Tester
def test_decorated_tester_hf_peft() -> None:
    def fn(
        mdl: PeftModel,
        test_loader: Dataset,
        extra_param_1: int,
        extra_param_2: float | None,
    ) -> TestResult:
        pass

    decorated = federate.tester.huggingface(fn)
    config: TesterSignatureSpec = getattr(decorated, "__fl_task_tester_config")
    assert config.net_parameter == "mdl"
    assert config.test_data_param == "test_loader"
    assert config.extra_test_kwargs == ["extra_param_1", "extra_param_2"]
    assert config.net_parameter_class_name == "PeftModel"


def test_decorated_tester_raises_invalid_return_type_hf_peft() -> None:
    def fn(
        mdl: PeftModel,
        test_loader: Dataset,
        extra_param_1: int,
        extra_param_2: float | None,
    ) -> Any:
        pass

    with pytest.raises(
        InvalidReturnType,
        match="Tester should return a fed_rag.types.TestResult or a subclass of it.",
    ):
        federate.tester.huggingface(fn)


def test_decorated_tester_fails_to_find_a_data_params_hf_peft() -> None:
    def fn(
        model: PeftModel,
    ) -> TestResult:
        pass

    msg = (
        "Inspection failed to find a data param for a test dataset."
        "For HuggingFace these params must be of type `datasets.Dataset`"
    )
    with pytest.raises(MissingDataParam, match=msg):
        federate.tester.huggingface(fn)


def test_decorated_tester_from_instance_method_hf_peft() -> None:
    class _TestClass:
        @federate.tester.huggingface
        def fn(
            self,
            mdl: PeftModel,
            test_loader: Dataset,
            extra_param_1: int,
        ) -> TestResult:
            pass

    obj = _TestClass()
    config: TesterSignatureSpec = getattr(obj.fn, "__fl_task_tester_config")
    assert config.net_parameter == "mdl"
    assert config.test_data_param == "test_loader"
    assert config.extra_test_kwargs == ["extra_param_1"]
    assert config.net_parameter_class_name == "PeftModel"


def test_decorated_tester_from_class_method_hf_peft() -> None:
    class _TestClass:
        @classmethod
        @federate.tester.huggingface
        def fn(
            cls,
            mdl: PeftModel,
            test_loader: Dataset,
            extra_param_1: int,
        ) -> TestResult:
            pass

    config: TesterSignatureSpec = getattr(
        _TestClass.fn, "__fl_task_tester_config"
    )
    assert config.net_parameter == "mdl"
    assert config.test_data_param == "test_loader"
    assert config.extra_test_kwargs == ["extra_param_1"]
    assert config.net_parameter_class_name == "PeftModel"
