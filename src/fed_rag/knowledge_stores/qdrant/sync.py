"""Qdrant Knowledge Store"""

from typing import TYPE_CHECKING, Any, Literal, Optional

from pydantic import Field, PrivateAttr, SecretStr

from fed_rag.base.knowledge_store import BaseKnowledgeStore
from fed_rag.exceptions import InvalidDistance, LoadNodeError
from fed_rag.knowledge_stores.qdrant.utils import check_qdrant_installed
from fed_rag.types.knowledge_node import KnowledgeNode

if TYPE_CHECKING:
    from qdrant_client import QdrantClient
    from qdrant_client.models import PointStruct


def _get_qdrant_client(
    host: str,
    port: int,
    ssl: bool = False,
    api_key: str | None = None,
    **kwargs: Any,
) -> "QdrantClient":
    """Get a QdrantClient

    NOTE: should be used within `QdrantKnowledgeStore` post validation that the
    qdrant extra has been installed.
    """
    from qdrant_client import QdrantClient

    if ssl:
        url = f"https://{host}:{port}"
    else:
        url = f"http://{host}:{port}"

    return QdrantClient(url=url, api_key=api_key, **kwargs)


def _covert_knowledge_node_to_qdrant_point(
    node: KnowledgeNode,
) -> "PointStruct":
    from qdrant_client.models import PointStruct

    return PointStruct(
        id=node.node_id, vector=node.embedding, payload=node.metadata
    )


class QdrantKnowledgeStore(BaseKnowledgeStore):
    """Qdrant Knowledge Store Class"""

    host: str = Field(default="localhost")
    port: int = Field(default=6333)
    ssl: bool = Field(default=False)
    api_key: SecretStr | None = Field(default=None)
    collection_name: str = Field(description="Name of Qdrant collection")
    collection_distance: Literal[
        "Cosine", "Euclid", "Dot", "Manhattan"
    ] = Field(
        description="Distance definition for collection", default="Cosine"
    )
    client_kwargs: dict[str, Any] = Field(default_factory=dict)
    _client: Optional["QdrantClient"] = PrivateAttr(default=None)

    def __init__(self, *args: Any, **kwargs: Any):
        check_qdrant_installed()
        super().__init__(*args, **kwargs)

    def _collection_exists(self, collection_name: str) -> bool:
        """Check if a collection exists."""
        return self.client.collection_exists(collection_name)  # type: ignore[no-any-return]

    def _create_collection(
        self, collection_name: str, vector_size: int, distance: str
    ) -> None:
        from qdrant_client.models import Distance, VectorParams

        try:
            # Try to convert to enum
            distance = Distance(distance)
        except ValueError:
            # Catch the ValueError from enum conversion and raise your custom error
            raise InvalidDistance(
                f"Unsupported distance: {distance}. "
                f"Mode must be one of: {', '.join([m.value for m in Distance])}"
            )

        self.client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=vector_size, distance=distance),
        )

    @property
    def client(self) -> "QdrantClient":
        if self._client is None:
            # get and set client
            self._client = _get_qdrant_client(
                host=self.host,
                port=self.port,
                ssl=self.ssl,
                api_key=self.api_key.get_secret_value()
                if self.api_key
                else None,
                **self.client_kwargs,
            )
        return self._client

    def load_node(self, node: KnowledgeNode) -> None:
        point = _covert_knowledge_node_to_qdrant_point(node)
        try:
            self.client.upsert(
                collection_name=self.collection_name, points=[point]
            )
        except Exception as e:
            raise LoadNodeError(
                f"Failed to load node {node.node_id} into collection '{self.collection_name}': {str(e)}"
            ) from e

    def load_nodes(self, nodes: list[KnowledgeNode]) -> None:
        if not nodes:
            return

        points = [_covert_knowledge_node_to_qdrant_point(n) for n in nodes]
        try:
            self.client.upload_points(
                collection_name=self.collection_name, points=points
            )
        except Exception as e:
            raise LoadNodeError(
                f"Loading nodes into collection '{self.collection}' failed"
            ) from e

    def retrieve(
        self, query_emb: list[float], top_k: int
    ) -> list[tuple[float, KnowledgeNode]]:
        raise NotImplementedError

    def delete_node(self, node_id: str) -> bool:
        raise NotImplementedError

    def clear(self) -> None:
        raise NotImplementedError

    @property
    def count(self) -> int:
        raise NotImplementedError

    def persist(self) -> None:
        raise NotImplementedError

    def load(self) -> None:
        raise NotImplementedError
