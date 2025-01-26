"""Base FL Task"""

from abc import abstractmethod, ABC
from typing import Any, Callable

from pydantic import BaseModel

from flwr.client.client import Client
from flwr.server.server import Server


class BaseFLTask(BaseModel, ABC):

    @property
    @abstractmethod
    def model(self) -> Any: ...

    @property
    @abstractmethod
    def training_loop(self) -> Callable: ...

    def simulate(self, num_clients, **kwargs) -> Any:
        """Simulate the FL task.

        Either use flwr's simulation tools, or create our own here.
        """
        ...

    def server(self) -> Server:
        """Create a flwr.Server object."""
        ...

    def client(self) -> Client:
        """Create a flwr.Client object."""
        ...
