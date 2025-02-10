"""Base FL Task"""

from abc import ABC, abstractmethod
from typing import Any, Callable

from flwr.client.client import Client
from flwr.server.server import Server
from pydantic import BaseModel, ConfigDict
from typing_extensions import Self

from fed_rag.exceptions import MissingFLTaskConfig


class BaseFLTaskConfig(BaseModel):
    pass


class BaseFLTask(BaseModel, ABC):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    @property
    @abstractmethod
    def training_loop(self) -> Callable:
        ...

    @classmethod
    @abstractmethod
    def from_configs(
        cls, trainer_cfg: BaseFLTaskConfig, tester_cfg: Any
    ) -> Self:
        ...

    @classmethod
    @abstractmethod
    def from_trainer_and_tester(
        cls, trainer: Callable, tester: Callable
    ) -> Self:
        try:
            trainer_cfg = getattr(trainer, "__fl_task_trainer_config")
        except AttributeError:
            msg = (
                "`__fl_task_trainer_config` has not been set on training loop. Make "
                "sure to decorate your training loop with the appropriate "
                "decorator."
            )
            raise MissingFLTaskConfig(msg)

        try:
            tester_cfg = getattr(tester, "__fl_task_tester_config")
        except AttributeError:
            msg = (
                "`__fl_task_tester_config` has not been set on tester callable. Make "
                "sure to decorate your tester with the appropriate decorator."
            )
            raise MissingFLTaskConfig(msg)
        return cls.from_configs(trainer_cfg, tester_cfg)

    @abstractmethod
    def simulate(self, num_clients: int, **kwargs: Any) -> Any:
        """Simulate the FL task.

        Either use flwr's simulation tools, or create our own here.
        """
        ...

    @abstractmethod
    def server(self, **kwargs: Any) -> Server:
        """Create a flwr.Server object."""
        ...

    @abstractmethod
    def client(self, **kwargs: Any) -> Client:
        """Create a flwr.Client object."""
        ...
