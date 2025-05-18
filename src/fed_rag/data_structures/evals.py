"""Data structures for fed_rag.evals"""

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
