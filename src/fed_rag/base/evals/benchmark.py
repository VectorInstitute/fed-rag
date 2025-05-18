"""Base Benchmark and Benchmarker"""

from abc import ABC, abstractmethod
from typing import Sequence

from pydantic import BaseModel, ConfigDict

from fed_rag import RAGSystem
from fed_rag.data_structures.evals import BenchmarkExample, BenchmarkResult


class BaseBenchmark(BaseModel, ABC):
    """Base Data Collator."""

    name: str
    examples: Sequence[BenchmarkExample]

    model_config = ConfigDict(arbitrary_types_allowed=True)


class BaseBenchmarker(BaseModel, ABC):
    rag_system: RAGSystem

    @abstractmethod
    def run(
        self,
        benchmark: BaseBenchmark,
        batch_size: int = 1,
        num_examples: int | None = None,
        num_workers: int = 1,
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
