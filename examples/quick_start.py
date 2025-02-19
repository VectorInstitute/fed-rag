import threading
import time
from typing import Any

# flwr
import flwr as fl

# torch
import torch
import torch.functional as F
import torch.nn as nn
import torchvision.transforms as transforms
from flwr.client import Client
from flwr.server import Server
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import IidPartitioner
from torch.types import Device
from torch.utils.data import DataLoader

# fedrag
from fed_rag.decorators import federate
from fed_rag.fl_tasks.pytorch import PyTorchFLTask
from fed_rag.types import TestResult, TrainResult

# partition cifar dataset
partitioner = IidPartitioner(num_partitions=2)
fds = FederatedDataset(
    dataset="uoft-cs/cifar10",
    partitioners={"train": partitioner},
)


def get_loaders(partition_id: int) -> tuple[DataLoader, DataLoader]:
    partition = fds.load_partition(partition_id)
    # Divide data on each node: 80% train, 20% test
    partition_train_test = partition.train_test_split(test_size=0.2, seed=42)
    pytorch_transforms = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )

    def apply_transforms(batch: torch.Tensor) -> torch.Tensor:
        """Apply transforms to the partition from FederatedDataset."""
        batch["img"] = [pytorch_transforms(img) for img in batch["img"]]
        return batch

    partition_train_test = partition_train_test.with_transform(
        apply_transforms
    )
    trainloader = DataLoader(
        partition_train_test["train"], batch_size=32, shuffle=True
    )
    testloader = DataLoader(partition_train_test["test"], batch_size=32)
    return trainloader, testloader


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


model = Net()


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
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
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
def test(
    m: torch.nn.Module, test_loader: DataLoader, device: Device
) -> TestResult:
    """My custom tester."""

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


# create your fl system
fl_task = PyTorchFLTask.from_trainer_and_tester(
    trainer=train_loop, tester=test
)


## What can you do with your fl system?

### 1. build a server

# requires the fl strategy and config
server = fl_task.server(model=model)

### 2. build a client trainer

# requires the exact same params as the defined trainer
clients = {}
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
    clients[i] = client
print(f"clients: {clients}", flush=True)


def start_local(
    event: threading.Event,
    client: Client | None = None,
    server: Server | None = None,
    **kwargs: Any,
) -> threading.Thread:
    """Spins up a server/client node for the given FLTask.

    Use threading.Event for graceful shutdown of nodes.
    """

    def node_wrapper(
        client: Client | None,
        server: Server | None,
        event: threading.Event,
        **kwargs: Any,
    ) -> None:
        if client:
            kwargs.update(client=client)
            start_fn = fl.client.start_client

        if server:
            kwargs.update(server=server)
            start_fn = fl.server.start_server

        # start node
        # try:
        start_fn(**kwargs)
        # except ValueError as e:
        #     pass

        # keep alive until event is set
        while not event.is_set():
            pass

        # TODO: graceful shutdown
        return

    # threads
    thread = threading.Thread(
        target=node_wrapper,
        args=(client, server, event),
        kwargs=kwargs,
    )
    thread.start()
    return thread


## start local
try:
    event = threading.Event()
    server_thread = start_local(
        event=event,
        server=server,
        server_address="[::]:8080",
    )
    time.sleep(5)
    client_0_thread = start_local(
        event=event, client=clients[0], server_address="[::]:8080"
    )
    client_1_thread = start_local(
        event=event, client=clients[1], server_address="[::]:8080"
    )

    server_thread.join()
    client_0_thread.join()
    client_1_thread.join()
except KeyboardInterrupt:
    event.set()
