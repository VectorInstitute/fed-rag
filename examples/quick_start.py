from typing import Any

import flwr as fl
import torch
from flwr.server.server_config import ServerConfig
from flwr.server.strategy import Strategy
from torch.utils.data import DataLoader

from fed_rag.decorators import federate
from fed_rag.fl_tasks.pytorch import PyTorchFLTask

# define your PyTorch model
model: torch.nn.Module = ...


# define your train loop, wrap it with @trainer decorator
@federate.trainer.pytorch
def train_loop(
    model: torch.nn.Module,
    train_data: DataLoader,
    val_data: DataLoader,
    num_epochs: int,
    learning_rate: float | None,
) -> Any:
    """My custom train loop."""
    pass


@federate.tester.pytorch
def test(models: torch.nn.Module, test_loader: DataLoader) -> Any:
    """My custom tester."""
    pass


# create your fl system
fl_task = PyTorchFLTask.from_trainer_and_tester(
    trainer=train_loop, tester=test
)


## What can you do with your fl system?

### 1. simulate a run
sim_results = fl_task.simulate(num_clients=2, strategy="fedavg")

### 2. build a server
strategy: Strategy = ...
config: ServerConfig = ...
server = fl_task.server(
    strategy=strategy,
    config=config,
    model=model,
)

if server:
    fl.server.start_server(
        server=fl_task.server,
        server_address="0.0.0.0:8080",
    )

### 3. build a client
client = fl_task.client(model=model)

if client:
    fl.client.start_client(
        server_address="0.0.0.0:8080", client=fl_task.client
    )

## BTW, you can still use your training loop as you would in centralized ML
train_data = ...
val_data = ...
num_epochs = ...
learning_rate = ...
train_loop(model, train_data, val_data, num_epochs, learning_rate)
