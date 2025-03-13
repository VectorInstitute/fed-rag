"""Decorators unit tests for HuggingFace"""

from typing import Callable

import pytest
from datasets import Dataset
from sentence_transformers import SentenceTransformer
from transformers import PreTrainedModel

from fed_rag.decorators import federate
from fed_rag.inspectors.huggingface import TrainerSignatureSpec
from fed_rag.types import TrainResult


def huggingface_pretrained_model_trainer(
    net: PreTrainedModel,
    train_dataset: Dataset,
    val_dataset: Dataset,
    extra_param_1: int,
    extra_param_2: float | None,
) -> TrainResult:
    pass


def huggingface_sentence_transformer_trainer(
    net: SentenceTransformer,
    train_dataset: Dataset,
    val_dataset: Dataset,
    extra_param_1: int,
    extra_param_2: float | None,
) -> TrainResult:
    pass


@pytest.mark.parametrize(
    "trainer_fn",
    [
        huggingface_pretrained_model_trainer,
        huggingface_sentence_transformer_trainer,
    ],
    ids=["pretrained_model", "sentence_transformer"],
)
def test_decorated_trainer(
    trainer_fn: Callable,
) -> None:
    decorated = federate.trainer.huggingface(trainer_fn)
    config: TrainerSignatureSpec = getattr(
        decorated, "__fl_task_trainer_config"
    )
    assert config.net_parameter == "net"
    assert config.train_data_param == "train_dataset"
    assert config.val_data_param == "val_dataset"
    assert config.extra_train_kwargs == ["extra_param_1", "extra_param_2"]
