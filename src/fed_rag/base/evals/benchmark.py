"""Base Benchmark and Benchmarker"""

from abc import ABC

from pydantic import BaseModel, ConfigDict

from fed_rag import RAGSystem


class BaseBenchmark(BaseModel, ABC):
    """Base Data Collator."""

    model_config = ConfigDict(arbitrary_types_allowed=True)


class BaseBenchmarker(BaseModel, ABC):
    rag_system: RAGSystem
