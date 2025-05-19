"""Base Benchmark and Benchmarker"""

from abc import ABC, abstractmethod
from typing import Any, Iterator, Sequence

from pydantic import BaseModel, ConfigDict, PrivateAttr, model_validator

from fed_rag import RAGSystem
from fed_rag.data_structures.evals import BenchmarkExample, BenchmarkResult


class BaseBenchmark(BaseModel, ABC):
    """Base Data Collator."""

    _examples: Sequence[BenchmarkExample] = PrivateAttr()

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @abstractmethod
    def _get_examples(self, **kwargs: Any) -> Sequence[BenchmarkExample]:
        """Method to get examples."""

    # give it a sequence interface for accessing examples more easily
    def __getitem__(self, index: int) -> BenchmarkExample:
        return self._examples.__getitem__(index)

    def __len__(self) -> int:
        return self._examples.__len__()

    # shouldn't override Pydantic BaseModels' __iter__
    def as_iterator(self) -> Iterator[BenchmarkExample]:
        return self._examples.__iter__()

    @model_validator(mode="after")
    def set_examples(self) -> "BaseBenchmark":
        self._examples = self._get_examples()
        return self


class BaseBenchmarker(BaseModel, ABC):
    rag_system: RAGSystem

    @abstractmethod
    def run(
        self,
        benchmark: BaseBenchmark,
        batch_size: int = 1,
        num_examples: int | None = None,
        num_workers: int = 1,
        **kwargs: Any,
    ) -> BenchmarkResult:
        """Execute the benchmark using the associated `RAGSystem`.

        Args:
            benchmark (BaseBenchmark): the benchmark to run the `RAGSystem` against.
            batch_size (int, optional): number of examples to process in a single batch.
            num_examples (int | None, optional): Number of examples to use from
                the benchmark. If None, then the entire collection of examples of
                the benchmark are ran. Defaults to None.
            num_workers (int, optional): concurrent execution via threads.

        Returns:
            BenchmarkResult: the benchmark result
        """
