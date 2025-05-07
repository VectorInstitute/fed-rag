"""Qdrant Knowledge Store"""

from typing import TYPE_CHECKING, Any, Optional

from pydantic import Field, PrivateAttr, SecretStr

from fed_rag.base.knowledge_store import BaseKnowledgeStore
from fed_rag.knowledge_stores.qdrant.utils import check_qdrant_installed
from fed_rag.types.knowledge_node import KnowledgeNode

if TYPE_CHECKING:
    from qdrant_client import QdrantClient


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


class QdrantKnowledgeStore(BaseKnowledgeStore):
    """Qdrant Knowledge Store Class"""

    host: str = Field(default="localhost")
    port: int = Field(default=6333)
    ssl: bool = Field(default=False)
    api_key: SecretStr | None = Field(default=None)
    client_kwargs: dict[str, Any] = Field(default_factory=dict)
    _client: Optional["QdrantClient"] = PrivateAttr(default=None)

    def __init__(self, *args: Any, **kwargs: Any):
        check_qdrant_installed()
        super().__init__(*args, **kwargs)

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
        raise NotImplementedError

    def load_nodes(self, nodes: list[KnowledgeNode]) -> None:
        raise NotImplementedError

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
