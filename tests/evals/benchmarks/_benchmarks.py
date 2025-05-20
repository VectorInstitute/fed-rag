from typing import Any, Generator, Sequence

from fed_rag.base.evals.benchmark import BaseBenchmark
from fed_rag.data_structures import BenchmarkExample
from fed_rag.evals.benchmarks.huggingface.mixin import (
    HuggingFaceBenchmarkMixin,
)


class TestBenchmark(BaseBenchmark):
    __test__ = (
        False  # needed for Pytest collision. Avoids PytestCollectionWarning
    )

    def _get_examples(self, **kwargs: Any) -> Sequence[BenchmarkExample]:
        return [
            BenchmarkExample(query="query 1", response="response 1"),
            BenchmarkExample(query="query 2", response="response 2"),
            BenchmarkExample(query="query 3", response="response 3"),
        ]

    def as_stream(self) -> Generator[BenchmarkExample, None, None]:
        for ex in self._get_examples():
            yield ex

    @property
    def num_examples(self) -> int:
        return len(self._get_examples())


class TestHFBenchmark(HuggingFaceBenchmarkMixin, BaseBenchmark):
    __test__ = (
        False  # needed for Pytest collision. Avoids PytestCollectionWarning
    )

    dataset_name = "test_benchmark"

    def _get_query_from_example(self, example: dict[str, Any]) -> str:
        return str(example["query"])

    def _get_response_from_example(self, example: dict[str, Any]) -> str:
        return str(example["response"])

    def _get_context_from_example(self, example: dict[str, Any]) -> str:
        return str(example["context"])


__all__ = ["TestBenchmark", "TestHFBenchmark"]
