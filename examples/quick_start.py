from typing import Any

import flwr as fl
import torch
from torch.utils.data import DataLoader

from fed_rag.decorators import federate
from fed_rag.fl_tasks.pytorch import PyTorchFLTask

# define your PyTorch model
model: torch.nn.Module = ...


# define your train loop, wrap it with @trainer decorator
@federate.pytorch
def train_loop(
    model: torch.nn.Module,
    train_data: DataLoader,
    val_data: DataLoader,
    num_epochs: int,
    learning_rate: float,
) -> Any:
    """My custom train loop."""
    pass


# create your fl system
fl_task = PyTorchFLTask.from_training_loop(train_loop)


## What can you do with your fl system?

### 1. simulate a run
sim_results = fl_task.simulate(num_clients=2, strategy="fedavg")

### 2. start server
fl.server.start_server(
    server=fl_task.server,
    server_address="0.0.0.0:8080",
)

### 3. start client
fl.client.start_client(server_address="0.0.0.0:8080", client=fl_task.client)
