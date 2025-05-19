"""Data structures for fed_rag.evals"""

from enum import Enum

from pydantic import BaseModel


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


class AggregationMode(str, Enum):
    """Mode for aggregating evaluation scores."""

    AVG = "avg"
    SUM = "sum"
    MAX = "max"
    MIN = "min"
