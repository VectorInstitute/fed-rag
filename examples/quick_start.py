import flwr as fl
import torch
from torch.utils.data import DataLoader

from fed_rag import create_fl_system_from_trainer
from fed_rag.decorators import trainer

model: torch.nn.Module = ...


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


fl_system = create_fl_system_from_trainer(trainer=train_loop)

# start server
fl.server.start_server(
    server=fl_system.server,
    server_address="0.0.0.0:8080",
)

# start client
fl.client.start_client(server_address="0.0.0.0:8080", client=fl_system.client)
