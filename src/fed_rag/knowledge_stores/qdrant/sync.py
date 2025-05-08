"""Qdrant Knowledge Store"""

from typing import TYPE_CHECKING, Any, Literal, Optional

from pydantic import Field, PrivateAttr, SecretStr, model_validator

from fed_rag.base.knowledge_store import BaseKnowledgeStore
from fed_rag.exceptions import (
    InvalidDistanceError,
    KnowledgeStoreError,
    KnowledgeStoreNotFoundError,
    LoadNodeError,
)
from fed_rag.knowledge_stores.qdrant.utils import check_qdrant_installed
from fed_rag.types.knowledge_node import KnowledgeNode

if TYPE_CHECKING:  # pragma: no cover
    from qdrant_client import QdrantClient
    from qdrant_client.http.models import ScoredPoint
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

    return QdrantClient(url=url, api_key=api_key, prefer_grpc=True, **kwargs)


def _convert_knowledge_node_to_qdrant_point(
    node: KnowledgeNode,
) -> "PointStruct":
    from qdrant_client.models import PointStruct

    return PointStruct(
        id=node.node_id, vector=node.embedding, payload=node.model_dump()
    )


def _convert_scored_point_to_knowledge_node_and_score_tuple(
    scored_point: "ScoredPoint",
) -> tuple[float, KnowledgeNode]:
    return (
        scored_point.score,
        KnowledgeNode.model_validate(scored_point.payload),
    )


class QdrantKnowledgeStore(BaseKnowledgeStore):
    """Qdrant Knowledge Store Class"""

    host: str = Field(default="localhost")
    port: int = Field(default=6334)
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

    def _collection_exists(self) -> bool:
        """Check if a collection exists."""
        return self.client.collection_exists(self.collection_name)  # type: ignore[no-any-return]

    def _create_collection(
        self, collection_name: str, vector_size: int, distance: str
    ) -> None:
        from qdrant_client.models import Distance, VectorParams

        try:
            # Try to convert to enum
            distance = Distance(distance)
        except ValueError:
            # Catch the ValueError from enum conversion and raise your custom error
            raise InvalidDistanceError(
                f"Unsupported distance: {distance}. "
                f"Mode must be one of: {', '.join([m.value for m in Distance])}"
            )

        try:
            self.client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(
                    size=vector_size, distance=distance
                ),
            )
        except Exception as e:
            raise KnowledgeStoreError(
                f"Failed to create collection: {str(e)}"
            ) from e

    def _ensure_collection_exists(self) -> None:
        if not self._collection_exists():
            raise KnowledgeStoreNotFoundError(
                f"Collection '{self.collection_name}' does not exist."
            )

    def _check_if_collection_exists_otherwise_create_one(
        self, vector_size: int
    ) -> None:
        if not self._collection_exists():
            try:
                self._create_collection(
                    collection_name=self.collection_name,
                    vector_size=vector_size,
                    distance=self.collection_distance,
                )
            except Exception:
                raise KnowledgeStoreError(
                    f"Failed to create new collection: {self.collection_name}"
                )

    @model_validator(mode="before")
    @classmethod
    def check_dependencies(cls, data: Any) -> Any:
        """Validate that qdrant dependencies are installed."""
        check_qdrant_installed()
        return data

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
        self._check_if_collection_exists_otherwise_create_one(
            vector_size=len(node.embedding)
        )

        point = _convert_knowledge_node_to_qdrant_point(node)
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

        self._check_if_collection_exists_otherwise_create_one(
            vector_size=len(nodes[0].embedding)
        )

        points = [_convert_knowledge_node_to_qdrant_point(n) for n in nodes]
        try:
            self.client.upload_points(
                collection_name=self.collection_name, points=points
            )
        except Exception as e:
            raise LoadNodeError(
                f"Loading nodes into collection '{self.collection_name}' failed: {str(e)}"
            ) from e

    def retrieve(
        self, query_emb: list[float], top_k: int
    ) -> list[tuple[float, KnowledgeNode]]:
        """Retrieve top-k nodes from the vector store."""
        from qdrant_client.conversions.common_types import QueryResponse

        self._ensure_collection_exists()

        try:
            hits: QueryResponse = self.client.query_points(
                collection_name=self.collection_name,
                query=query_emb,
                limit=top_k,
            )
        except Exception as e:
            raise KnowledgeStoreError(
                f"Failed to retrieve from collection '{self.collection_name}': {str(e)}"
            ) from e

        return [
            _convert_scored_point_to_knowledge_node_and_score_tuple(pt)
            for pt in hits.points
        ]

    def delete_node(self, node_id: str) -> bool:
        """Delete a node based on its node_id."""
        from qdrant_client.http.models import (
            FieldCondition,
            Filter,
            MatchValue,
            UpdateResult,
            UpdateStatus,
        )

        self._ensure_collection_exists()

        try:
            res: UpdateResult = self.client.delete(
                collection_name=self.collection_name,
                points_selector=Filter(
                    must=[
                        FieldCondition(
                            key="node_id", match=MatchValue(value=node_id)
                        )
                    ]
                ),
            )
        except Exception:
            raise KnowledgeStoreError(
                f"Failed to delete node: '{node_id}' from collection '{self.collection_name}'"
            )

        return bool(res.status == UpdateStatus.COMPLETED)

    def clear(self) -> None:
        self._ensure_collection_exists()

        # delete the collection
        try:
            self.client.delete_collection(collection_name=self.collection_name)
        except Exception as e:
            raise KnowledgeStoreError(
                f"Failed to delete collection '{self.collection_name}': {str(e)}"
            ) from e

    @property
    def count(self) -> int:
        from qdrant_client.http.models import CollectionInfo

        self._ensure_collection_exists()

        try:
            collection_info: CollectionInfo = self.client.get_collection(
                collection_name=self.collection_name
            )
        except Exception as e:
            raise KnowledgeStoreError(
                f"Failed to get vector count for collection '{self.collection_name}': {str(e)}"
            ) from e

        if collection_info.vectors_count is None:
            raise KnowledgeStoreError(
                "Collection exists, but `vectors_count` is None."
            )
        else:
            return int(collection_info.vectors_count)

    def persist(self) -> None:
        """Persist a knowledge store to disk."""
        raise NotImplementedError(
            "`persist()` is not available in QdrantKnowledgeStore."
        )

    def load(self) -> None:
        """Load a previously persisted knowledge store."""
        raise NotImplementedError(
            "`load()` is not available in QdrantKnowledgeStore. "
            "Data is automatically persisted and loaded from the Qdrant server."
        )
