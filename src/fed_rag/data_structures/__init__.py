"""
fed_rag.data_structures

Only components defined in `__all__` are considered stable and public.
"""

from .bridge import BridgeMetadata, CompatibleVersions
from .evals import (
    AggregationMode,
    BenchmarkEvaluatedExample,
    BenchmarkExample,
    BenchmarkResult,
)
from .generator import Context, Prompt, Query
from .knowledge_node import KnowledgeNode, NodeContent, NodeType
from .rag import RAGConfig, RAGResponse, SourceNode
from .results import TestResult, TrainResult

__all__ = [
    # bridge
    "BridgeMetadata",
    "CompatibleVersions",
    # evals
    "AggregationMode",
    "BenchmarkExample",
    "BenchmarkResult",
    "BenchmarkEvaluatedExample",
    # generator
    "Query",
    "Context",
    "Prompt",
    # results
    "TrainResult",
    "TestResult",
    # knowledge node
    "KnowledgeNode",
    "NodeType",
    "NodeContent",
    # rag
    "RAGConfig",
    "RAGResponse",
    "SourceNode",
]
