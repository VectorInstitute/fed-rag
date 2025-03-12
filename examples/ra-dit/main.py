"""Federate RAG via RA-DIT."""

# torch
import torch

# hf
from sentence_transformers import InputExample, SentenceTransformer, losses
from torch.types import Device
from torch.utils.data import DataLoader

# fedrag
from fed_rag.types import TrainResult

from .rag_system import main as get_rag_system

model_name = "/model-weights/Llama-2-7b-hf"
rag_system = get_rag_system(model_name)

# Define your train examples. You need more than just two examples...
train_examples = [
    InputExample(texts=["My first sentence", "My second sentence"], label=0.8),
    InputExample(texts=["Another pair", "Unrelated sentence"], label=0.3),
]

# Define your train dataset, the dataloader and the train loss
train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=16)


def retriever_train_loop(
    model: SentenceTransformer,
    train_data: DataLoader,
    val_data: DataLoader,
    device: Device,
    num_epochs: int,
    learning_rate: float | None,
) -> TrainResult:
    model.to(device)
    # dummy loss for now
    model.train()
    train_loss = losses.CosineSimilarityLoss(model)

    # Tune the model
    model.fit(
        train_objectives=[(train_data, train_loss)],
        epochs=num_epochs,
        optimizer_params={"lr": learning_rate},
        warmup_steps=100,
    )
    return TrainResult(train_loss)


if __name__ == "__main__":
    retriever = rag_system.retriever
    train_result = retriever_train_loop(
        model=retriever.query_encoder,
        train_data=train_dataloader,
        val_data=train_dataloader,
        device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
        num_epochs=1,
        learning_rate=0.1,
    )
    print(train_result)
