"""Federate RAG via RA-DIT."""

# torch

# hf
from datasets import Dataset
from torch.types import Device
from transformers import PreTrainedModel, Trainer
from transformers.trainer_utils import TrainOutput

# fedrag
from fed_rag.decorators import federate
from fed_rag.types import TestResult, TrainResult

from .rag_system import main as get_rag_system

model_name = "/model-weights/Llama-2-7b-hf"
# model_name = "meta-llama/Llama-2-7b-hf"
rag_system = get_rag_system(model_name)


@federate.trainer.huggingface
def generator_train_loop(
    model: PreTrainedModel,
    train_data: Dataset,
    val_data: Dataset,
    device: Device,
) -> TrainResult:
    del val_data

    model.to(device)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=val_data,
        compute_metrics=compute_metrics,
    )

    train_output: TrainOutput = trainer.train()

    return TrainResult(loss=train_output.training_loss)


@federate.tester.huggingface
def retriever_evaluate(m: PreTrainedModel, test_data: Dataset) -> TestResult:
    return TestResult(loss=42.0, metrics={})
