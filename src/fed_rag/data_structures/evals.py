"""Data structures for fed_rag.evals"""

from enum import Enum

from pydantic import BaseModel

from .rag import RAGResponse


class BenchmarkExample(BaseModel):
    """Benchmark example data class."""

    query: str
    response: str
    context: str | None = None


class BenchmarkResult(BaseModel):
    """Benchmark result data class."""

    score: float
    metric_name: str
    num_examples_used: int
    num_total_examples: int
    evaluations_file: str | None


class BenchmarkEvaluatedExample(BaseModel):
    """Evaluated benchmark example data class."""

    score: float
    example: BenchmarkExample
    rag_response: RAGResponse

    def model_dump_json_without_embeddings(self) -> str:
        return self.model_dump_json(
            exclude={
                "rag_response": {
                    "source_nodes": {"__all__": {"node": {"embedding"}}}
                }
            }
        )


class AggregationMode(str, Enum):
    """Mode for aggregating evaluation scores."""

    AVG = "avg"
    SUM = "sum"
    MAX = "max"
    MIN = "min"
