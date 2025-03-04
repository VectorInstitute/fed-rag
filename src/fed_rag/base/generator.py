"""Base Generator"""

from abc import ABC, abstractmethod

from pydantic import BaseModel, ConfigDict


class BaseGenerator(BaseModel, ABC):
    """Base Generator Class."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @abstractmethod
    def generate(self, input: str) -> str:
        """Generate an output from a given input."""
        ...
