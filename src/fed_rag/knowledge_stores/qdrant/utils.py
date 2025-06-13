"""Qdrant utils module."""

from importlib.util import find_spec
from typing import TYPE_CHECKING

from fed_rag.data_structures.knowledge_node import KnowledgeNode
from fed_rag.exceptions import KnowledgeStoreError, MissingExtraError

if TYPE_CHECKING:  # pragma: no cover
    from qdrant_client.http.models import ScoredPoint
    from qdrant_client.models import PointStruct


def check_qdrant_installed() -> None:
    if find_spec("qdrant_client") is None:
        raise MissingExtraError(
            "Qdrant knowledge stores require the qdrant-client to be installed. "
            "To fix please run `pip install fed-rag[qdrant]`."
        )


def convert_knowledge_node_to_qdrant_point(
    node: KnowledgeNode,
) -> "PointStruct":
    from qdrant_client.models import PointStruct

    if node.embedding is None:
        raise KnowledgeStoreError(
            "Cannot load a node with embedding set to None."
        )

    return PointStruct(
        id=node.node_id,
        vector=node.embedding,
        payload=node.model_dump_without_embeddings(),
    )


def convert_scored_point_to_knowledge_node_and_score_tuple(
    scored_point: "ScoredPoint",
) -> tuple[float, KnowledgeNode]:
    knowledge_data = scored_point.payload
    knowledge_data.update(
        embedding=scored_point.vector
    )  # attach vector to embedding if it is even returned
    return (
        scored_point.score,
        KnowledgeNode.model_validate(knowledge_data),
    )
