"""Base Benchmark and Benchmarker"""

from typing import Any

from pydantic import BaseModel

from fed_rag import RAGSystem
from fed_rag.base.evals.benchmark import BaseBenchmark
from fed_rag.base.evals.metric import BaseEvaluationMetric
from fed_rag.data_structures.evals import BenchmarkResult


class Benchmarker(BaseModel):
    rag_system: RAGSystem

    def run(
        self,
        benchmark: BaseBenchmark,
        metric: BaseEvaluationMetric,
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

        if num_examples:
            examples_iterator = iter(benchmark[:num_examples])
        else:
            examples_iterator = benchmark.as_iterator()

        scores = []
        num_seen = 0
        for example in examples_iterator:
            result = self.rag_system.query(example.query)
            score = metric(prediction=result.response, actual=example.response)
            scores.append(score)
            num_seen += 1

        final_score = metric.aggregate_fn(scores)
        return BenchmarkResult(
            score=final_score,
            metric_name=metric.name,
            num_examples_used=num_seen,
            num_total_examples=len(benchmark),
        )
