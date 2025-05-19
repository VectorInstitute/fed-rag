"""HuggingFaceBenchmarkMixin"""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, ClassVar

from pydantic import BaseModel, PrivateAttr, model_validator

if TYPE_CHECKING:  # pragma: no cover
    from datasets import Dataset


class HuggingFaceBenchmarkMixin(BaseModel, ABC):
    """Mixin for HuggingFace Benchmarks"""

    dataset_name: ClassVar[str]
    configuration_name: str | None = None
    split: str = "test"
    streaming: bool = False
    load_kwargs: dict[str, Any] = {}
    _dataset: "Dataset" = PrivateAttr()

    @abstractmethod
    def _map_dataset_example(self, example: dict[str, Any]) -> dict[str, Any]:
        """Map the examples in the dataset to include a `~fed_rag.data_structures.evals.BenchmarkExample`."""

    @model_validator(mode="after")
    def _load_dataset(self) -> "HuggingFaceBenchmarkMixin":
        from datasets import load_dataset

        self._dataset = load_dataset(
            self.dataset_name,
            name=self.configuration_name,
            split=self.split,
            streaming=self.streaming,
            **self.load_kwargs,
        )

        return self
