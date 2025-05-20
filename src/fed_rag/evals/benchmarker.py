"""Base Benchmark and Benchmarker"""

import contextlib
from typing import Any, Generator

from pydantic import BaseModel
from typing_extensions import assert_never

from fed_rag import RAGSystem
from fed_rag.base.evals.benchmark import BaseBenchmark
from fed_rag.base.evals.metric import BaseEvaluationMetric
from fed_rag.data_structures.evals import (
    AggregationMode,
    BenchmarkExample,
    BenchmarkResult,
)


class Benchmarker(BaseModel):
    """Benchmarker"""

    rag_system: RAGSystem

    def _update_running_score(
        self,
        agg: AggregationMode,
        running_score: float | None,
        next_score: float,
        num_examples_seen: int,
    ) -> float:
        """Update the running score.

        Args:
            agg (AggregationMode): aggregation mode.
            running_score (float): the running score to be updated.
            next_score (float): the score of the latest scored example.
            num_examples_seen (int): the number of examples seen prior to the
                latest scored example.

        Returns:
            float: the updated running score
        """
        if not running_score:
            return next_score

        match agg:
            case AggregationMode.AVG:
                return (num_examples_seen * running_score + next_score) / (
                    num_examples_seen + 1
                )
            case AggregationMode.SUM:
                return running_score + next_score
            case AggregationMode.MAX:
                if running_score < next_score:
                    return next_score
                else:
                    return running_score
            case AggregationMode.MIN:
                if running_score > next_score:
                    return next_score
                else:
                    return running_score
            case _:  # pragma: no cover
                assert_never(agg)

    @contextlib.contextmanager
    def _get_examples_iterator(
        self, benchmark: BaseBenchmark, is_streaming: bool
    ) -> Generator[BenchmarkExample, None, None]:
        """Wrapper over the iterator or stream.

        To handle generator clean up safely.
        """
        if is_streaming:
            examples_iterator = benchmark.as_stream()
        else:
            examples_iterator = benchmark.as_iterator()

        try:
            yield examples_iterator
        finally:
            if hasattr(examples_iterator, "close"):
                examples_iterator.close()

    def run(
        self,
        benchmark: BaseBenchmark,
        metric: BaseEvaluationMetric,
        is_streaming: bool = False,
        agg: AggregationMode | str = "avg",
        batch_size: int = 1,
        num_examples: int | None = None,
        num_workers: int = 1,
        **kwargs: Any,
    ) -> BenchmarkResult:
        """Execute the benchmark using the associated `RAGSystem`.

        Args:
            agg (AggregationMode | str): the aggregation mode to apply to all example scores.
                Modes include `avg`, `sum`, `max`, or `min`.
            benchmark (BaseBenchmark): the benchmark to run the `RAGSystem` against.
            batch_size (int, optional): number of examples to process in a single batch.
            metric (BaseEvaluationMetric): the metric to use for evaluation.
            num_examples (int | None, optional): Number of examples to use from
                the benchmark. If None, then the entire collection of examples of
                the benchmark are ran. Defaults to None.
            num_workers (int, optional): concurrent execution via threads.

        Returns:
            BenchmarkResult: the benchmark result

        TODO: implement concurrent as well as batch execution. Need RAGSystem
        to be able to handle batches as well.
        """

        with self._get_examples_iterator(
            benchmark, is_streaming
        ) as examples_iterator:
            running_score = None
            num_seen = 0
            for example in examples_iterator:
                if num_seen == num_examples:
                    break

                # prediction
                result = self.rag_system.query(example.query)

                # evaluation
                score = metric(
                    prediction=result.response, actual=example.response
                )

                # update running score
                running_score = self._update_running_score(
                    agg=agg,
                    running_score=running_score,
                    next_score=score,
                    num_examples_seen=num_seen,
                )

                num_seen += 1

        return BenchmarkResult(
            score=running_score,
            metric_name=metric.__class__.__name__,
            num_examples_used=num_seen,
            num_total_examples=benchmark.num_examples,
        )
