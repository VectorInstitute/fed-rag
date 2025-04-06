import uuid
from abc import ABC, abstractmethod

from pydantic import BaseModel, Field
from typing_extensions import Self


def generate_ks_id() -> str:
    return str(uuid.uuid4())


class ManagedMixin(BaseModel, ABC):
    ks_id: str = Field(default_factory=generate_ks_id)

    @classmethod
    @abstractmethod
    def from_name_and_id(cls, ks_id: str) -> Self:
        """Load a managed Knowledge Store by id."""
