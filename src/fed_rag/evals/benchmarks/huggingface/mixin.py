"""HuggingFaceBenchmarkMixin"""

from abc import ABC, abstractmethod
from typing import (
    TYPE_CHECKING,
    Any,
    ClassVar,
    Generator,
    Optional,
    Sequence,
    Union,
    cast,
)

from pydantic import BaseModel, PrivateAttr

from fed_rag.data_structures.evals import BenchmarkExample
from fed_rag.exceptions import EvalsError

if TYPE_CHECKING:  # pragma: no cover
    from datasets import Dataset, IterableDataset


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
        query = self._get_query_from_example(example)
        response = self._get_response_from_example(example)
        context = self._get_context_from_example(example)

        example[BENCHMARK_EXAMPLE_JSON_KEY] = {
            "query": query,
            "response": response,
            "context": context,
        }
        return example

    def _load_dataset(self) -> Union["Dataset", "IterableDataset"]:
        from datasets import load_dataset

        try:
            loaded_dataset = load_dataset(
                self.dataset_name,
                name=self.configuration_name,
                split=self.split,
                streaming=self.streaming,
                **self.load_kwargs,
            )
        except Exception as e:
            raise EvalsError(
                f"Failed to load dataset, `{self.dataset_name}`, due to error: {str(e)}"
            ) from e

        # add BenchmarkExample to dataset
        return loaded_dataset.map(self._map_dataset_example)

    @property
    def dataset(self) -> Union["Dataset", "IterableDataset"]:
        if self._dataset is None:
            self._dataset = self._load_dataset()

        return self._dataset

    # Provide required implementations for abstractmethods in ~BaseBenchmark
    def _get_examples(self, **kwargs: Any) -> Sequence[BenchmarkExample]:
        from datasets import Dataset

        if isinstance(self.dataset, Dataset):
            return [
                BenchmarkExample.model_validate(el)
                for el in self.dataset[BENCHMARK_EXAMPLE_JSON_KEY]
            ]
        else:
            return []  # examples should be streamed

    def as_stream(self) -> Generator[BenchmarkExample, None, None]:
        from datasets import IterableDataset

        # check if dataset is an iterable one
        if isinstance(self.dataset, IterableDataset):
            iterable_dataset = self.dataset
        else:
            iterable_dataset = self.dataset.to_iterable_dataset()

        # map the iterable_dataset to get the examples i
        iterable_dataset = iterable_dataset.map(self._map_dataset_example)

        for hf_example in iterable_dataset:
            yield BenchmarkExample.model_validate(
                hf_example[BENCHMARK_EXAMPLE_JSON_KEY]
            )

    @property
    def num_examples(self) -> int:
        from datasets import SplitInfo

        if splits := self.dataset.info.splits:
            try:
                split_info = splits[self.split]
                split_info = cast(
                    SplitInfo,
                    split_info,
                )
                return int(split_info.num_examples)
            except (KeyError, TypeError):
                raise EvalsError(
                    f"Unable to get size of dataset: `{self.dataset_name}`. "
                    f"Split, `{self.split}` does not exist in the splits of the dataset."
                )
        else:
            raise EvalsError(
                f"Unable to get size of dataset: `{self.dataset_name}`. "
                "The dataset does not have any listed splits."
            )
