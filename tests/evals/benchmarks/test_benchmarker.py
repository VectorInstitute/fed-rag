from typing import Any

from fed_rag import RAGSystem
from fed_rag.base.evals.metric import BaseEvaluationMetric
from fed_rag.evals.benchmarker import Benchmarker

from . import _benchmarks as benchmarks


class MyMetric(BaseEvaluationMetric):
    answers: list[float] = [1, 2, 3]
    ix: int = -1

    def __call__(
        self, prediction: str, actual: str, *args: Any, **kwargs: Any
    ) -> float:
        self.ix = self.ix + 1
        return self.answers[self.ix]


def test_benchmarker_avg(mock_rag_system: RAGSystem) -> None:
    # arrange
    test_benchmark = benchmarks.TestBenchmark()
    benchmarker = Benchmarker(rag_system=mock_rag_system)
    metric = MyMetric()

    # act
    result = benchmarker.run(
        benchmark=test_benchmark, metric=metric, agg="avg"
    )

    # assert
    assert result.score == 2
    assert result.num_examples_used == 3
    assert result.num_total_examples == 3
    assert result.metric_name == "MyMetric"


def test_benchmarker_sum(mock_rag_system: RAGSystem) -> None:
    # arrange
    test_benchmark = benchmarks.TestBenchmark()
    benchmarker = Benchmarker(rag_system=mock_rag_system)
    metric = MyMetric()

    # act
    result = benchmarker.run(
        benchmark=test_benchmark, metric=metric, agg="sum"
    )

    # assert
    assert result.score == 6
    assert result.num_examples_used == 3
    assert result.num_total_examples == 3
    assert result.metric_name == "MyMetric"


def test_benchmarker_min(mock_rag_system: RAGSystem) -> None:
    # arrange
    test_benchmark = benchmarks.TestBenchmark()
    benchmarker = Benchmarker(rag_system=mock_rag_system)
    metric = MyMetric()

    # act
    result = benchmarker.run(
        benchmark=test_benchmark, metric=metric, agg="min"
    )

    # assert
    assert result.score == 1
    assert result.num_examples_used == 3
    assert result.num_total_examples == 3
    assert result.metric_name == "MyMetric"


def test_benchmarker_max(mock_rag_system: RAGSystem) -> None:
    # arrange
    test_benchmark = benchmarks.TestBenchmark()
    benchmarker = Benchmarker(rag_system=mock_rag_system)
    metric = MyMetric()

    # act
    result = benchmarker.run(
        benchmark=test_benchmark, metric=metric, agg="max"
    )

    # assert
    assert result.score == 3
    assert result.num_examples_used == 3
    assert result.num_total_examples == 3
    assert result.metric_name == "MyMetric"


def test_benchmarker_num_examples(mock_rag_system: RAGSystem) -> None:
    # arrange
    test_benchmark = benchmarks.TestBenchmark()
    benchmarker = Benchmarker(rag_system=mock_rag_system)
    metric = MyMetric()

    # act
    result = benchmarker.run(
        benchmark=test_benchmark, metric=metric, num_examples=2
    )

    # assert
    assert result.score == 1.5
    assert result.num_examples_used == 2
    assert result.num_total_examples == 3
    assert result.metric_name == "MyMetric"


def test_benchmarker_min_reversed(mock_rag_system: RAGSystem) -> None:
    # arrange
    test_benchmark = benchmarks.TestBenchmark()
    benchmarker = Benchmarker(rag_system=mock_rag_system)
    metric = MyMetric()
    metric.answers = metric.answers[::-1]

    # act
    result = benchmarker.run(
        benchmark=test_benchmark, metric=metric, agg="min"
    )

    # assert
    assert result.score == 1
    assert result.num_examples_used == 3
    assert result.num_total_examples == 3
    assert result.metric_name == "MyMetric"


def test_benchmarker_max_reversed(mock_rag_system: RAGSystem) -> None:
    # arrange
    test_benchmark = benchmarks.TestBenchmark()
    benchmarker = Benchmarker(rag_system=mock_rag_system)
    metric = MyMetric()
    metric.answers = metric.answers[::-1]

    # act
    result = benchmarker.run(
        benchmark=test_benchmark, metric=metric, agg="max"
    )

    # assert
    assert result.score == 3
    assert result.num_examples_used == 3
    assert result.num_total_examples == 3
    assert result.metric_name == "MyMetric"


def test_benchmarker_with_streaming_avg(mock_rag_system: RAGSystem) -> None:
    # arrange
    test_benchmark = benchmarks.TestBenchmark()
    benchmarker = Benchmarker(rag_system=mock_rag_system)
    metric = MyMetric()

    # act
    result = benchmarker.run(
        benchmark=test_benchmark, metric=metric, agg="avg", is_streaming=True
    )

    # assert
    assert result.score == 2
    assert result.num_examples_used == 3
    assert result.num_total_examples == 3
    assert result.metric_name == "MyMetric"
