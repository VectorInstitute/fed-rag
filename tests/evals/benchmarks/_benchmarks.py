from typing import Any, Sequence

from fed_rag.base.evals.benchmark import BaseBenchmark
from fed_rag.data_structures import BenchmarkExample


class MyBenchmark(BaseBenchmark):
    def _get_examples(self, **kwargs: Any) -> Sequence[BenchmarkExample]:
        return [
            BenchmarkExample(query="query 1", response="response 1"),
            BenchmarkExample(query="query 2", response="response 2"),
            BenchmarkExample(query="query 3", response="response 3"),
        ]


__all__ = ["MyBenchmark"]
