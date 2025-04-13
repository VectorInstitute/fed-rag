"""Base Benchmark."""

import logging
import time
from abc import ABC, abstractmethod
from concurrent.futures import Future, ThreadPoolExecutor
from threading import Lock
from typing import Any

import pandas as pd
from pydantic import BaseModel, ConfigDict, PrivateAttr
from typing_extensions import Self
from accelerate import PartialState
from accelerate.utils import gather_object


from fed_rag.types.rag_system import RAGSystem


class BenchmarkResult(BaseModel):
    score: float


class ExamplePred(BaseModel):
    pred: str


class ScoredExamplePred(ExamplePred):
    score: float

    @classmethod
    def from_example_pred(cls, pred: ExamplePred, score: float) -> Self:
        return cls(pred=pred.pred, score=score)


class BaseBenchmark(BaseModel, ABC):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    examples: pd.DataFrame
    generate_prompt_template: str | None
    _logger: logging.Logger = PrivateAttr()

    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self._logger = logging.getLogger(f"ra_dit.benchmark.{self.__class__.__name__}")

    @property
    def logger(self) -> logging.Logger:
        return self._logger

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
        start_time = time.time()
        distributed_state = PartialState()
        self._update_prompt_template(rag_system=rag_system)
        tasks_total = len(self.examples)
        log_interval = max(1, tasks_total // 10)

        # Calculate chunk for this process
        chunk_size = tasks_total // distributed_state.num_processes
        start_idx = distributed_state.process_index * chunk_size
        end_idx = start_idx + chunk_size
        # Last process takes any remainder
        if distributed_state.process_index == distributed_state.num_processes - 1:
            end_idx = tasks_total

        self.logger.info(
            f"Process {distributed_state.process_index} handling rows {start_idx} to {end_idx-1}"
        )

        examples_list = [e for _, e in self.examples.iterrows()]

        def process_example(example: pd.Series) -> ScoredExamplePred:
            pred = self._predict_example(example=example, rag_system=rag_system)
            return self._evaluate_prediction(example=example, pred=pred)

        with distributed_state.split_between_processes(examples_list) as local_examples:

            # Process the local examples
            local_scored_examples = []
            for i, example in enumerate(local_examples):
                scored_example = process_example(example=example)
                local_scored_examples.append(
                    {"pred": scored_example.pred, "score": scored_example.score}
                )
                if (i + 1) % log_interval == 0:
                    self.logger.debug(
                        f"Process {distributed_state.process_index}: Processed {i + 1} / {len(local_examples)} examples"
                    )

        distributed_state.wait_for_everyone()
        all_scored_examples = gather_object(local_scored_examples)

        if distributed_state.is_main_process:
            scored_examples = [
                item for sublist in all_scored_examples for item in sublist
            ]
            self.logger.debug(f"Example: {scored_examples}")
            agg_score = self.aggregate_example_scores(scored_examples=scored_examples)
            self.logger.info("Successfully ran benchmark.")
            self.logger.info(f"Benchmark took {time.time() - start_time:.2f} seconds")
            return BenchmarkResult(score=agg_score)
        else:
            return None
