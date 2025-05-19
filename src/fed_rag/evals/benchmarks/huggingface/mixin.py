"""HuggingFaceBenchmarkMixin"""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, ClassVar, Optional, Sequence

from pydantic import BaseModel, PrivateAttr

from fed_rag.data_structures.evals import BenchmarkExample

if TYPE_CHECKING:  # pragma: no cover
    from datasets import Dataset


BENCHMARK_EXAMPLE_JSON_KEY = "__benchmark_example_json"


class HuggingFaceBenchmarkMixin(BaseModel, ABC):
    """Mixin for HuggingFace Benchmarks"""

    dataset_name: ClassVar[str]
    configuration_name: str | None = None
    split: str = "test"
    streaming: bool = False
    load_kwargs: dict[str, Any] = {}
    _dataset: Optional["Dataset"] = PrivateAttr(default=None)

    @abstractmethod
    def _get_query_from_example(self, example: dict[str, Any]) -> str:
        """Derive the query from the example."""

    @abstractmethod
    def _get_response_from_example(self, example: dict[str, Any]) -> str:
        """Derive the response from the example."""

    @abstractmethod
    def _get_context_from_example(self, example: dict[str, Any]) -> str | None:
        """Derive the context from the example."""

    def _map_dataset_example(self, example: dict[str, Any]) -> dict[str, Any]:
        """Map the examples in the dataset to include a `~fed_rag.data_structures.evals.BenchmarkExample`."""
        query = self._get_context_from_example(example)
        response = self._get_response_from_example(example)
        context = self._get_context_from_example(example)

        example[BENCHMARK_EXAMPLE_JSON_KEY] = {
            "query": query,
            "response": response,
            "context": context,
        }
        return example

    def _load_dataset(self) -> "Dataset":
        from datasets import load_dataset

        loaded_dataset = load_dataset(
            self.dataset_name,
            name=self.configuration_name,
            split=self.split,
            streaming=self.streaming,
            **self.load_kwargs,
        )

        # add BenchmarkExample to dataset
        return loaded_dataset.map(
            self._map_dataset_example, cache_file_name=None
        )

    @property
    def dataset(self) -> "Dataset":
        if self._dataset is None:
            self._dataset = self._load_dataset()

        return self._dataset

    def _get_examples(self, **kwargs: Any) -> Sequence[BenchmarkExample]:
        return [
            BenchmarkExample.model_validate(el)
            for el in self.dataset[BENCHMARK_EXAMPLE_JSON_KEY]
        ]
