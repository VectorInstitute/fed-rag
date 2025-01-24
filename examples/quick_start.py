import flwr as fl
import torch
from torch.utils.data import DataLoader

from fed_rag import create_fl_system_from_trainer
from fed_rag.decorators import trainer

# define your PyTorch model
model: torch.nn.Module = ...


# define your train loop, wrap it with @trainer decorator
@trainer
def train_loop(
    model: torch.nn.Module,
    train_data: DataLoader,
    val_data: DataLoader,
    num_epochs: int,
    learning_rate: float,
):
    """My custom train loop."""
    ...


# create your fl system
fl_system = create_fl_system_from_trainer(trainer=train_loop)


## What can you do with your fl system?

### 1. simulate a run
sim_results = fl_system.simulate(num_clients=2, strategy="fedavg")

### 2. start server
fl.server.start_server(
    server=fl_system.server,
    server_address="0.0.0.0:8080",
)

### 3. start client
fl.client.start_client(server_address="0.0.0.0:8080", client=fl_system.client)
