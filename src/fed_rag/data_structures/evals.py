"""Data structures for fed_rag.evals"""

from enum import Enum

from pydantic import BaseModel

from .rag import RAGResponse


class BenchmarkExample(BaseModel):
    """Benchmark example data class.

    This class represents a single benchmark example.

    Attributes:
        query(str): Query string.
        response(str): Response string.
        context(str|None): Context string.
    """

    query: str
    response: str
    context: str | None = None


class BenchmarkResult(BaseModel):
    """Benchmark result data class.

    This class represents the result of a benchmark.

    Attributes:
        score(float): Score of the benchmark example.
        metric_name(str): Name of the metric used for scoring.
        num_examples_used(int): Number of examples used for scoring.
        num_total_examples(int): Number of total examples in the benchmark.
        evaluations_file(str|None): Path to the evaluations file.

    """

    score: float
    metric_name: str
    num_examples_used: int
    num_total_examples: int
    evaluations_file: str | None


class BenchmarkEvaluatedExample(BaseModel):
    """Evaluated benchmark example data class.

    This class represents an evaluated benchmark example.

    Attributes:
        score(float): Score of the benchmark example.
        example(BenchmarkExample): Benchmark example.
        rag_response(RAGResponse): RAG response.

    """

    score: float
    example: BenchmarkExample
    rag_response: RAGResponse

    def model_dump_json_without_embeddings(self) -> str:
        """
        Generates and returns a JSON representation of the model excluding specific
        embedding-related data.
        """

        return self.model_dump_json(
            exclude={
                "rag_response": {
                    "source_nodes": {"__all__": {"node": {"embedding"}}}
                }
            }
        )


class AggregationMode(str, Enum):
    """Mode for aggregating evaluation scores.

    This enum defines the available modes for aggregating multiple evaluation scores
    into a single value.

    Attributes:
        AVG: Calculates the arithmetic mean of the scores.
        SUM: Calculates the sum of all scores.
        MAX: Takes the maximum score value.
        MIN: Takes the minimum score value.
    """

    AVG = "avg"
    SUM = "sum"
    MAX = "max"
    MIN = "min"
