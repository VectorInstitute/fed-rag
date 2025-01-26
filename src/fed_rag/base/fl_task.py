"""Base FL Task"""

from abc import ABC, abstractmethod
from typing import Any, Callable

from flwr.client.client import Client
from flwr.server.server import Server
from pydantic import BaseModel
from typing_extensions import Self

from fed_rag.exceptions import MissingFLTaskConfig


class BaseFLTaskConfig(BaseModel):
    pass


class BaseFLTask(BaseModel, ABC):
    @property
    @abstractmethod
    def net(self) -> Any:
        ...

    @property
    @abstractmethod
    def training_loop(self) -> Callable:
        ...

    @classmethod
    @abstractmethod
    def from_config(cls, cfg: BaseFLTaskConfig) -> Self:
        ...

    @classmethod
    @abstractmethod
    def from_training_loop(cls, training_loop: Callable) -> Self:
        cfg = getattr(training_loop, "__fl_task_config", None)
        if not cfg:
            msg = (
                "`__fl_task_config` has not been set on training loop. Make "
                "sure to decorate your training loop with the appropriate "
                "decorator."
            )
            raise MissingFLTaskConfig(msg)
        return cls.from_config(cfg)

    def simulate(self, num_clients: int, **kwargs: Any) -> Any:
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
