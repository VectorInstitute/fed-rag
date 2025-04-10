"""Base Benchmark."""

from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import pandas as pd
from pydantic import BaseModel, ConfigDict

from fed_rag.types.rag_system import RAGSystem


class BenchmarkResult(BaseModel):
    score: float


class ExamplePred(BaseModel):
    pred: Any
    score: float | None = None


class BaseBenchmark(BaseModel, ABC):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    examples: pd.DataFrame

    @abstractmethod
    def _predict_example(
        self, example: pd.Series, rag_system: RAGSystem
    ) -> ExamplePred:
        """Logic for predicting an example with a given 'rag_system'."""

    @abstractmethod
    def _evaluate_prediction(
        self, example: pd.Series, pred: ExamplePred
    ) -> ExamplePred:
        """Evaluate a prediction."""

    @abstractmethod
    def _aggregate_example_scores(
        self, score_examples: list[ExamplePred]
    ) -> float:
        """Custom aggregated example scores.

        Subclasses implement this method.
        """

    def aggregate_example_scores(
        self, score_examples: list[ExamplePred]
    ) -> float:
        """Aggregated example scores"""
        if not all(e.score is not None for e in score_examples):
            raise ValueError("Not all examples were scored.")
        return self._aggregate_example_scores(score_examples=score_examples)

    def run(
        self, rag_system: RAGSystem, num_threads: int = 1
    ) -> BenchmarkResult:
        """Run the benchmark with the given rag_system."""

        def process_example(example: ExamplePred) -> ExamplePred:
            pred = self._predict_example(
                example=example, rag_system=rag_system
            )
            return self._evaluate_prediction(example=example, pred=pred)

        if num_threads > 1:
            with ThreadPoolExecutor(max_workers=num_threads) as executor:
                scored_examples = list(
                    executor.map(
                        process_example,
                        [e for _, e in self.examples.iterrows()],
                    )
                )
        else:
            scored_examples = []
            for _, example in self.examples.iterrows():
                scored_examples.append(process_example(example=example))
        agg_score = self.aggregate_example_scores(
            score_examples=scored_examples
        )
        return BenchmarkResult(score=agg_score)
