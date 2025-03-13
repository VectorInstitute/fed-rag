"""Federate RAG via RA-DIT."""

# torch
import torch

# hf
from datasets import Dataset
from sentence_transformers import (
    SentenceTransformer,
    SentenceTransformerTrainer,
    losses,
)
from torch.types import Device
from torch.utils.data import DataLoader
from transformers.trainer_utils import TrainOutput

# fedrag
from fed_rag.decorators import federate
from fed_rag.types import TestResult, TrainResult

from .rag_system import main as get_rag_system

# model_name = "/model-weights/Llama-2-7b-hf"
model_name = "meta-llama/Llama-2-7b-hf"
rag_system = get_rag_system(model_name)

# Define your train examples. You need more than just two examples...
train_dataset = Dataset.from_dict(
    {
        "text1": ["My first sentence", "Another pair"],
        "text2": ["My second sentence", "Unrelated sentence"],
        "label": [0.8, 0.3],
    }
)


# Define your train dataset, the dataloader and the train loss
train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=16)


@federate.trainer.pytorch
def retriever_train_loop(
    model: SentenceTransformer,
    train_data: DataLoader,
    val_data: DataLoader,
    device: Device,
    train_dataset: Dataset,
) -> TrainResult:
    del val_data

    model.to(device)
    loss = losses.CosineSimilarityLoss(model)

    trainer = SentenceTransformerTrainer(
        model=model,
        train_dataset=train_dataset,
        loss=loss,
        # args=args,
        # eval_dataset=eval_dataset,
        # evaluator=dev_evaluator,
    )
    train_output: TrainOutput = trainer.train()

    return TrainResult(loss=train_output.training_loss)


@federate.tester.pytorch
def retriever_evaluate(
    m: SentenceTransformer, test_loader: DataLoader
) -> TestResult:
    return TestResult(loss=42.0, metrics={})


if __name__ == "__main__":
    retriever = rag_system.retriever
    train_result = retriever_train_loop(
        model=retriever.query_encoder,
        train_data=train_dataloader,
        val_data=train_dataloader,
        device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
        train_dataset=train_dataset,
    )
    print(train_result)
