from typing import Any, Sequence

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


class TestHFBenchmark(HuggingFaceBenchmarkMixin, BaseBenchmark):
    __test__ = (
        False  # needed for Pytest collision. Avoids PytestCollectionWarning
    )

    dataset_name = "nerdai/_test_rag_dataset"

    def _get_examples(self, **kwargs: Any) -> Sequence[BenchmarkExample]:
        return [
            BenchmarkExample(query="query 1", response="response 1"),
            BenchmarkExample(query="query 2", response="response 2"),
            BenchmarkExample(query="query 3", response="response 3"),
        ]

    def _map_dataset_example(self, example: dict) -> dict[str, Any]:
        raise NotImplementedError


__all__ = ["TestBenchmark", "TestHFBenchmark"]
