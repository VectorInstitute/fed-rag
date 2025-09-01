"""Auxiliary types for RAG System"""

from typing import Any

from PIL import Image
from pydantic import BaseModel, ConfigDict

from .knowledge_node import KnowledgeNode


class SourceNode(BaseModel):
    """
    Represents a source node with an associated score and a reference to a knowledge node.

    This class is used to associate a score with a specific knowledge node, allowing
    for the evaluation or ranking of knowledge entities. It provides access to the
    underlying node's attributes and functionalities through a convenient getattr wrapper.

    Attributes:
        score (float): The score associated with the node, representing some metric
            of evaluation or importance.
        node (KnowledgeNode): The knowledge node this source node is referencing.
    """

    score: float
    node: KnowledgeNode

    def __getattr__(self, __name: str) -> Any:
        """Convenient wrapper on getattr of associated node."""
        return getattr(self.node, __name)


class RAGResponse(BaseModel):
    """
    Represents a response object for a Retrieval-Augmented Generation (RAG)
    system.

    This class is used to encapsulate information about the response including
    the generated response text, optional raw response details, and the source
    nodes leveraged in generating the response.

    Attributes:
        response (str): The generated response text from the RAG system.
        raw_response (str | None): Optional raw response details, if available.
        source_nodes (list[SourceNode]): A list of source nodes that were used
            to generate the response, each with an associated score and reference
            to a knowledge node.


    """

    response: str
    raw_response: str | None = None
    source_nodes: list[SourceNode]

    def __str__(self) -> str:
        return self.response


class RAGConfig(BaseModel):
    """
    Configuration class for a Retrieval-Augmented Generation (RAG) model.

    RAGConfig stores configuration parameters used to define behavior and
    constraints for the RAG model, including retrieval result limits and token
    separation for contexts.

    Attributes:
        top_k: int
            The maximum number of retrieved documents to consider.
        context_separator: str
            The string used to separate contexts during text generation.
    """

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
