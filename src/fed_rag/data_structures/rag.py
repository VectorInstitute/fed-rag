"""Auxiliary types for RAG System"""

from typing import Any

from pydantic import BaseModel

from .knowledge_node import KnowledgeNode


class SourceNode(BaseModel):
    score: float
    node: KnowledgeNode

    def __getattr__(self, __name: str) -> Any:
        """Convenient wrapper on getattr of associated node."""
        return getattr(self.node, __name)


class RAGResponse(BaseModel):
    response: str
    raw_response: str | None = None
    source_nodes: list[SourceNode]

    def __str__(self) -> str:
        return self.response


class RAGConfig(BaseModel):
    top_k: int
    context_separator: str = "\n"
