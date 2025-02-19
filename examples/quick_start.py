import threading
from typing import Any

import flwr as fl
import torch
import torch.functional as F
import torch.nn as nn
from flwr.client import Client
from flwr.server import Server
from flwr.server.server_config import ServerConfig
from flwr.server.strategy import Strategy
from torch.types import Device
from torch.utils.data import DataLoader

from fed_rag.decorators import federate
from fed_rag.fl_tasks.pytorch import PyTorchFLTask
from fed_rag.types import TestResult, TrainResult


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
    return TrainResult(loss=0.1)


@federate.tester.pytorch
def test(m: torch.nn.Module, test_loader: DataLoader) -> TestResult:
    """My custom tester."""
    return TestResult(loss=0.0, metrics={})


# create your fl system
fl_task = PyTorchFLTask.from_trainer_and_tester(
    trainer=train_loop, tester=test
)


## What can you do with your fl system?

### 1. build a server

# requires the fl strategy and config
strategy: Strategy = ...
config: ServerConfig = ...

server = fl_task.server(
    strategy=strategy,
    config=config,
)

if server:
    fl.server.start_server(
        server=fl_task.server,
        server_address="0.0.0.0:8080",
    )

### 2. build a client trainer

# requires the exact same params as the defined trainer
train_data = ...
val_data = ...
device = ...
num_epochs = ...
learning_rate = ...

client = fl_task.client(
    # train params
    model=model,
    train_data=train_data,
    val_data=val_data,
    device=device,
    num_epochs=num_epochs,
    learning_rate=learning_rate,
)

if client:
    fl.client.start_client(
        server_address="0.0.0.0:8080", client=fl_task.client
    )


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
        start_fn(**kwargs)

        # keep alive until event is set
        while not event.is_set():
            continue

        # TODO: graceful shutdown
        return

    # threads
    thread = threading.Thread(
        target=node_wrapper,
        args=(fl_task, client, server, event),
        kwargs=kwargs,
    )
    thread.start()
    return thread


## BTW, you can still use your training loop as you would in centralized ML
train_data = ...
val_data = ...
num_epochs = ...
learning_rate = ...
device = ...
train_loop(model, train_data, val_data, device, num_epochs, learning_rate)
