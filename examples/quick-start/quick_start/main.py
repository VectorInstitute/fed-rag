# for quick cli
from typing import Literal

# torch
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.types import Device
from torch.utils.data import DataLoader

# fedrag
from fed_rag.decorators import federate
from fed_rag.fl_tasks.pytorch import PyTorchFLTask
from fed_rag.types import TestResult, TrainResult

# cifar data for this quickstart example
from ._cifar_dataloaders import get_loaders


# define your PyTorch model
class Net(torch.nn.Module):
    """Model (simple CNN adapted from 'PyTorch: A 60 Minute Blitz')"""

    def __init__(self) -> None:
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


# define your train loop, wrap it with @trainer decorator
@federate.trainer.pytorch
def train_loop(
    model: torch.nn.Module,
    train_data: DataLoader,
    val_data: DataLoader,
    device: Device,
    num_epochs: int,
    learning_rate: float | None,
) -> TrainResult:
    """My custom train loop."""

    model.to(device)  # move model to GPU if available
    criterion = torch.nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.SGD(
        model.parameters(), lr=learning_rate, momentum=0.9
    )
    model.train()
    running_loss = 0.0
    for _ in range(num_epochs):
        for batch in train_data:
            images = batch["img"]
            labels = batch["label"]
            optimizer.zero_grad()
            loss = criterion(model(images.to(device)), labels.to(device))
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

    avg_trainloss = running_loss / len(train_data)
    return TrainResult(loss=avg_trainloss)


@federate.tester.pytorch
def test(m: torch.nn.Module, test_loader: DataLoader) -> TestResult:
    """My custom tester."""

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    m.to(device)
    criterion = torch.nn.CrossEntropyLoss()
    correct, loss = 0, 0.0
    with torch.no_grad():
        for batch in test_loader:
            images = batch["img"].to(device)
            labels = batch["label"].to(device)
            outputs = m(images)
            loss += criterion(outputs, labels).item()
            correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()
    accuracy = correct / len(test_loader.dataset)
    return TestResult(loss=loss, metrics={"accuracy": accuracy})


# Create your FLTask
fl_task = PyTorchFLTask.from_trainer_and_tester(
    trainer=train_loop, tester=test
)

## What can you do with your FLTask?

### 1. construct a server
model = Net()  # lora weights
server = fl_task.server(model=model)

### 2. construct a client trainer
clients = []
for i in range(2):
    train_data, val_data = get_loaders(partition_id=i)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    num_epochs = 1
    learning_rate = 0.1

    client = fl_task.client(
        # train params
        model=model,
        train_data=train_data,
        val_data=val_data,
        device=device,
        num_epochs=num_epochs,
        learning_rate=learning_rate,
    )
    clients.append(client)


# NOTE: The code below is merely for building a quick CLI to start server, and clients.
def start_component(
    component: Literal["server", "client_1", "client_2"]
) -> None:
    """For starting any of the FL Task components."""
    import flwr as fl

    if component == "server":
        fl.server.start_server(server=server, server_address="[::]:8080")
    elif component == "client_1":
        fl.client.start_client(client=clients[0], server_address="[::]:8080")
    elif component == "client_2":
        fl.client.start_client(client=clients[1], server_address="[::]:8080")
    else:
        raise ValueError("Unrecognized component.")


if __name__ == "__main__":
    import fire

    fire.Fire(start_component)
