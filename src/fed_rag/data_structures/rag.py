"""Auxiliary types for RAG System"""

from typing import Any

from PIL import Image
from pydantic import BaseModel, ConfigDict

from .knowledge_node import KnowledgeNode


class SourceNode(BaseModel):
    score: float
    node: KnowledgeNode

    def __getattr__(self, __name: str) -> Any:
        """Convenient wrapper on getattr of associated node."""
        return getattr(self.node, __name)


class RAGResponse(BaseModel):
    """Response class returned by querying RAG systems."""

    response: str
    raw_response: str | None = None
    source_nodes: list[SourceNode]

    def __str__(self) -> str:
        return self.response


class RAGConfig(BaseModel):
    top_k: int
    context_separator: str = "\n"


class _MultiModalDataContainer(BaseModel):
    """A multi-modal data container.

    This class represents a multimodal representation of a RAG query.

    Attributes:
        text: Text content of query.
        images: Images content of query.
        audios: Audios content of query.
        videos: Videos content of query.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)
    text: str
    images: list[Image.Image] | None = None
    audios: list[Any] | None = None
    videos: list[Any] | None = None

    def __str__(self) -> str:
        return self.text


class Query(_MultiModalDataContainer):
    """Query data structure.

    This class represents a multimodal representation of a RAG query.

    Attributes:
        text: Text content of query.
        images: Images content of query.
        audios: Audios content of query.
        videos: Videos content of query.
    """

    pass


class Context(_MultiModalDataContainer):
    """Context data structure.

    This class represents a multimodal representation of RAG context.

    Attributes:
        text: Text content of query.
        images: Images content of query.
        audios: Audios content of query.
        videos: Videos content of query.
    """

    pass


class Prompt(_MultiModalDataContainer):
    """Prompt data structure.

    This class represents a multimodal representation of a prompt given to a
    multi-modal LLM.

    Attributes:
        text: Text content of query.
        images: Images content of query.
        audios: Audios content of query.
        videos: Videos content of query.
    """

    pass
