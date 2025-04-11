"""Base Benchmark."""

import logging
from abc import ABC, abstractmethod
from concurrent.futures import Future, ThreadPoolExecutor
from threading import Lock
from typing import Any

import pandas as pd
from pydantic import BaseModel, ConfigDict
from typing_extensions import Self

from fed_rag.types.rag_system import RAGSystem


class BenchmarkResult(BaseModel):
    score: float


class ExamplePred(BaseModel):
    pred: Any

    def __str__(self):
        return str(self.pred)


class ScoredExamplePred(ExamplePred):
    score: float

    @classmethod
    def from_example_pred(cls, pred: ExamplePred, score: float) -> Self:
        return cls(pred=pred.pred, score=score)


class BaseBenchmark(BaseModel, ABC):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    examples: pd.DataFrame
    generate_prompt_template: str | None
    logger: logging.Logger = logging.getLogger("ra_dit.benchmark")

    @property
    @abstractmethod
    def name(self) -> str:
        """Name of benchmark."""

    @abstractmethod
    def _predict_example(
        self, example: pd.Series, rag_system: RAGSystem
    ) -> ExamplePred:
        """Logic for predicting an example with a given 'rag_system'."""

    @abstractmethod
    def _evaluate_prediction(
        self, example: pd.Series, pred: ExamplePred
    ) -> ScoredExamplePred:
        """Evaluate a prediction."""

    @abstractmethod
    def _aggregate_example_scores(
        self, scored_examples: list[ScoredExamplePred]
    ) -> float:
        """Custom aggregated example scores.

        Subclasses implement this method.
        """

    def aggregate_example_scores(
        self, scored_examples: list[ScoredExamplePred]
    ) -> float:
        """Aggregated example scores"""
        return self._aggregate_example_scores(scored_examples=scored_examples)

    def _update_prompt_template(self, rag_system: RAGSystem) -> None:
        if self.generate_prompt_template and hasattr(
            rag_system.generator, "prompt_template"
        ):
            rag_system.generator.prompt_template = self.generate_prompt_template

    def run(self, rag_system: RAGSystem, num_threads: int = 1) -> BenchmarkResult:
        """Run the benchmark with the given rag_system."""
        self.logger.info(
            f"Running benchmark {self.name} with num_threads: {num_threads}"
        )

        self._update_prompt_template(rag_system=rag_system)

        def process_example(example: pd.Series) -> ScoredExamplePred:
            pred = self._predict_example(example=example, rag_system=rag_system)
            return self._evaluate_prediction(example=example, pred=pred)

        def progress_indicator(future: Future):
            global lock, tasks_total, tasks_completed
            with lock:
                tasks_completed += 1
                self.logger.debug(
                    f"{tasks_completed}/{tasks_total} completed, {tasks_total-tasks_completed} remain."
                )

        tasks_total = len(self.examples)
        if num_threads > 1:
            lock = Lock()
            tasks_completed = 0
            with ThreadPoolExecutor(max_workers=num_threads) as executor:
                scored_examples = list(
                    executor.map(
                        process_example,
                        [e for _, e in self.examples.iterrows()],
                    )
                )
                for future in scored_examples:
                    future.add_done_callback(progress_indicator)
        else:
            log_interval = max(1, tasks_total // 10)
            scored_examples = []
            for ix, example in self.examples.iterrows():
                scored_examples.append(process_example(example=example))
                if (ix + 1) % log_interval == 0:
                    self.logger.debug(f"Processed {ix + 1} / {tasks_total} examples")

        agg_score = self.aggregate_example_scores(scored_examples=scored_examples)
        return BenchmarkResult(score=agg_score)
