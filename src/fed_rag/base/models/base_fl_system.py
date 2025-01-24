"""Base Federated System"""

from abc import abstractmethod

from flwr.client.client import Client
from flwr.server.server import Server
from pydantic import BaseModel


class FLSystemConfig(BaseModel):
    pass


class BaseFLSystem(BaseModel):
    fl_config: FLSystemConfig

    @abstractmethod
    def client(self) -> Client:
        ...

    @property
    @abstractmethod
    def server(self) -> Server:
        ...
