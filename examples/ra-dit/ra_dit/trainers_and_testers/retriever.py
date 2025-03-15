"""Retriever trainer and tester following RA-DIT."""

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
from transformers.trainer_utils import TrainOutput

# fedrag
from fed_rag.decorators import federate
from fed_rag.types import TestResult, TrainResult

# Define your train examples. You need more than just two examples...
train_dataset = Dataset.from_dict(
    {
        "text1": ["My first sentence", "Another pair"],
        "text2": ["My second sentence", "Unrelated sentence"],
        "label": [0.8, 0.3],
    }
)

val_dataset = Dataset.from_dict(
    {
        "text1": ["My third sentence"],
        "text2": ["My fourth sentence"],
        "label": [0.42],
    }
)


@federate.trainer.huggingface
def retriever_train_loop(
    model: SentenceTransformer,
    train_data: Dataset,
    val_data: Dataset,
    device: Device,
) -> TrainResult:
    del val_data

    model.to(device)
    loss = losses.CosineSimilarityLoss(model)

    trainer = SentenceTransformerTrainer(
        model=model,
        train_dataset=train_data,
        loss=loss,
        # args=args,
        # eval_dataset=eval_dataset,
        # evaluator=dev_evaluator,
    )
    train_output: TrainOutput = trainer.train()

    return TrainResult(loss=train_output.training_loss)


@federate.tester.huggingface
def retriever_evaluate(
    model: SentenceTransformer, test_data: Dataset
) -> TestResult:
    return TestResult(loss=42.0, metrics={})


if __name__ == "__main__":
    """centralized"""
    from ra_dit.retrievers.dragon import retriever

    train_result = retriever_train_loop(
        model=retriever.query_encoder,
        train_data=train_dataset,
        val_data=val_dataset,
        device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
    )
    print(train_result)
