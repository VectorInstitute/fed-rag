"""Federate RAG via RA-DIT."""

from typing import Any

# torch
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.types import Device
from torch.utils.data import DataLoader, Dataset

# hf
from sentence_transformers import SentenceTransformer

# fedrag
from fed_rag.decorators import federate
from fed_rag.fl_tasks.pytorch import PyTorchFLTask
from fed_rag.types import TestResult, TrainResult
from fed_rag.retrievers.hf_sentence_transformer import HFSentenceTransformerRetriever


from .rag_system import main as get_rag_system

model_name = "/model-weights/Llama-2-7b-hf"
rag_system = get_rag_system(model_name)


class SentencesDataset(Dataset):

    def __init__(self, sentences=list[str]):
        self.sentences = sentences

    def __getitem__(self, index):
        return self.sentences[index]

    def __len__(self):
        return len(self.sentences)


# build custom Dataset
sentences = ["this is a test", "this is also a test", "this is yet another test"]
dataset = SentencesDataset(sentences)


def custom_collate_fn(batch: Any) -> dict:
    sentences = [s for s in batch]
    return {"sentences": sentences}


data_loader = DataLoader(dataset, batch_size=3, collate_fn=custom_collate_fn)


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
    criterion = torch.nn.MSELoss().to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
    model.train()
    running_loss = 0.0
    for _ in range(num_epochs):
        for batch in train_data:
            sentences = batch["sentences"]
            outputs: torch.Tensor = torch.from_numpy(model.encode(sentences))
            targets = torch.zeros_like(outputs)
            optimizer.zero_grad()
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

    avg_trainloss = running_loss / len(train_data)
    return TrainResult(loss=avg_trainloss)


if __name__ == "__main__":

    retriever = rag_system.retriever
    train_result = retriever_train_loop(
        model=retriever.query_encoder,
        train_data=data_loader,
        val_data=data_loader,
        device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
        num_epochs=1,
        learning_rate=0.1,
    )
    print(train_result)
