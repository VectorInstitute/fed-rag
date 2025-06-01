"""MCP Knowledge Store"""

import asyncio
from typing import Callable

from pydantic import Field
from typing_extensions import Self

from fed_rag.base.no_encode_knowledge_store import (
    BaseAsyncNoEncodeKnowledgeStore,
)
from fed_rag.data_structures import KnowledgeNode
from fed_rag.exceptions import MCPKnowledgeStoreError

from .sources.base import BaseMCPKnowledgeSource

DEFAULT_SCORE = 1.0
DEFAULT_KNOWLEDGE_STORE_NAME = "default-mcp"
DEFAULT_TOP_K = 2


class MCPKnowledgeStore(BaseAsyncNoEncodeKnowledgeStore):
    """MCP Knowledge Store.

    Retrieve knowledge from attached MCP servers.
    """

    name: str = DEFAULT_KNOWLEDGE_STORE_NAME
    sources: dict[str, BaseMCPKnowledgeSource]
    reranker_callback: (
        Callable[[list[KnowledgeNode], str], list[tuple[float, KnowledgeNode]]]
        | None
    ) = Field(
        description="Custom callback for applying re-ranking on retrieved nodes from MPC sources.",
        default=None,
    )

    def __init__(
        self,
        sources: list[BaseMCPKnowledgeSource] | None = None,
        reranker_callback: Callable | None = None,
    ):
        sources = sources or []
        sources_dict = {s.name: s for s in sources}
        super().__init__(
            sources=sources_dict, reranker_callback=reranker_callback
        )

    def add_source(self, source: BaseMCPKnowledgeSource) -> Self:
        """Add a source to knowledge store.

        Support fluent chaining.
        """

        if not isinstance(source, BaseMCPKnowledgeSource):
            raise MCPKnowledgeStoreError(
                f"Cannot add source of type: {type(source)}"
            )

        if source.name in self.sources:
            raise MCPKnowledgeStoreError(
                f"A source with the same name, {source.name}, already exists."
            )

        self.sources[source.name] = source
        return self

    def with_reranker(self, reranker_fn: Callable) -> Self:
        """Setter for reranker_callback.

        For convenience and users who prefer the fluent style.
        """
        self.reranker_callback = reranker_fn
        return self

    async def _retrieve_from_source(
        self, query: str, source_id: str
    ) -> list[KnowledgeNode]:
        source = self.sources[source_id]
        call_tool_result = await source.retrieve(query)
        return source.call_tool_result_to_knowledge_nodes_list(
            call_tool_result
        )

    async def retrieve(
        self, query: str, top_k: int = DEFAULT_TOP_K
    ) -> list[tuple[float, KnowledgeNode]]:
        """Retrieve from all MCP knowledge sources.

        Queries all attached MCP sources concurrently and returns scored knowledge nodes.
        If a reranker_callback is provided, it will be used to score and rank the results.
        Otherwise, all nodes receive a default score and are limited by top_k.

        Args:
            query (str): query to send to each MCP source
            top_k (int): number of nodes to retrieve

        Returns:
            List of (score, KnowledgeNode) tuples, sorted by relevance score (highest first)

        Example:
            Basic usage without reranking:

            >>> store = MCPKnowledgeStore()
            >>> results = await store.retrieve("What's the weather in Tokyo?", top_k=3)
            >>> for score, node in results:
            ...     print(f"Score: {score}, Content: {node.text_content[:50]}...")

            With custom reranker:

            >>> def simple_reranker(knowledge_nodes, query):
            ...     # Score based on query word overlap
            ...     query_words = set(query.lower().split())
            ...     scored = []
            ...     for node in knowledge_nodes:
            ...         text_words = set(node.text_content.lower().split())
            ...         overlap = len(query_words & text_words)
            ...         score = overlap / len(query_words) if query_words else 0
            ...         scored.append((score, node))
            ...     return sorted(scored, key=lambda x: x[0], reverse=True)
            ...
            >>> store.with_reranker(simple_reranker)
            >>> results = await store.retrieve("machine learning algorithms", top_k=5)
            >>> # Returns nodes ranked by keyword overlap, not limited to top_k
        """
        tasks = []
        for source_id in self.sources:
            tasks.append(self._retrieve_from_source(query, source_id))

        all_node_lists = await asyncio.gather(*tasks)
        # flatten nested list
        knowledge_nodes = [
            node for node_list in all_node_lists for node in node_list
        ]

        if self.reranker_callback:
            # user can supply their own re-ranker here
            reranked = self.reranker_callback(knowledge_nodes, query)
            return reranked[:top_k]

        return [(DEFAULT_SCORE, node) for node in knowledge_nodes[:top_k]]

    # Not implemented methods
    async def load_node(self, node: KnowledgeNode) -> None:
        raise NotImplementedError(
            "load_node is not implemented for MCPKnowledgeStore."
        )

    async def load_nodes(self, nodes: list[KnowledgeNode]) -> None:
        raise NotImplementedError(
            "load_nodes is not implemented for MCPKnowledgeStore."
        )

    async def delete_node(self, node_id: str) -> None:
        raise NotImplementedError(
            "delete_node is not implemented for MCPKnowledgeStore."
        )

    async def clear(self) -> None:
        raise NotImplementedError(
            "clear is not implemented for MCPKnowledgeStore."
        )

    @property
    def count(self) -> int:
        raise NotImplementedError(
            "count is not implemented for MCPKnowledgeStore."
        )

    def persist(self) -> None:
        raise NotImplementedError(
            "persist is not implemented for MCPKnowledgeStore."
        )

    def load(self) -> None:
        raise NotImplementedError(
            "load is not implemented for MCPKnowledgeStore."
        )
