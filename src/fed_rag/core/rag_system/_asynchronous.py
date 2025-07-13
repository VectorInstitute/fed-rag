"""Internal Async RAG System Module"""

import asyncio
from typing import TYPE_CHECKING

import torch
from pydantic import BaseModel, ConfigDict

from fed_rag.base.bridge import BridgeRegistryMixin
from fed_rag.data_structures import RAGConfig, RAGResponse, SourceNode
from fed_rag.data_structures.rag import Query
from fed_rag.exceptions import RAGSystemError

if TYPE_CHECKING:  # pragma: no cover
    # to avoid circular imports, using forward refs
    from fed_rag.base.generator import BaseGenerator
    from fed_rag.base.knowledge_store import BaseAsyncKnowledgeStore
    from fed_rag.base.retriever import BaseRetriever


class _AsyncRAGSystem(BridgeRegistryMixin, BaseModel):
    """Unbridged implementation of AsyncRAGSystem.

    IMPORTANT: This is an internal implementation class.
    It should only be used by bridge mixins and never referenced directly
    by user code or other parts of the library.

    All interaction with RAG systems should be through the public AsyncRAGSystem class.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)
    generator: "BaseGenerator"
    retriever: "BaseRetriever"
    knowledge_store: "BaseAsyncKnowledgeStore"
    rag_config: RAGConfig

    async def query(self, query: str | Query) -> RAGResponse:
        """Query the RAG system."""
        source_nodes = await self.retrieve(query)
        context = self._format_context(source_nodes)
        response = await self.generate(query=query, context=context)
        return RAGResponse(source_nodes=source_nodes, response=response)

    async def batch_query(
        self, queries: list[str | Query]
    ) -> list[RAGResponse]:
        """Batch query the RAG system."""
        source_nodes_list = await self.batch_retrieve(queries)
        contexts = [
            self._format_context(source_nodes)
            for source_nodes in source_nodes_list
        ]
        responses = await self.batch_generate(queries, contexts)
        return [
            RAGResponse(source_nodes=source_nodes, response=response)
            for source_nodes, response in zip(source_nodes_list, responses)
        ]

    async def retrieve(self, query: str | Query) -> list[SourceNode]:
        """Retrieve from multiple collections based on query modalities."""
        # Get multimodal embeddings from retriever
        query_emb_tensor = self.retriever.encode_query(query)

        # Convert to separate embeddings by modality
        modality_embeddings = self._prepare_modality_embeddings(
            query_emb_tensor, query
        )

        # Retrieve from each modality collection concurrently
        retrieve_tasks = []
        for modality, embedding in modality_embeddings.items():
            if embedding is not None:
                task = self.knowledge_store.retrieve_by_modality(
                    modality=modality,
                    query_emb=embedding,
                    top_k=self.rag_config.top_k,
                )
                retrieve_tasks.append((modality, task))

        # Wait for all retrievals to complete
        all_results = []
        for modality, task in retrieve_tasks:
            modality_results = await task
            for score, node in modality_results:
                source_node = SourceNode(score=score, node=node)
                source_node.modality = modality
                all_results.append(source_node)

        all_results.sort(key=lambda x: x.score, reverse=True)
        return all_results[: self.rag_config.top_k]

    async def batch_retrieve(
        self, queries: list[str | Query]
    ) -> list[list[SourceNode]]:
        """Batch retrieve from multiple collections."""
        retrieve_tasks = [self.retrieve(query) for query in queries]
        return await asyncio.gather(*retrieve_tasks)

    def _prepare_modality_embeddings(
        self, embedding_tensor: torch.Tensor, query: str | Query
    ) -> dict[str, list[float]]:
        """Extract embeddings for each modality present in the query."""
        modality_embeddings = {}

        if isinstance(query, str):
            # Text-only query
            modality_embeddings["text"] = embedding_tensor.squeeze().tolist()
        elif isinstance(query, Query):
            # Check what modalities are present in the query
            available_modalities = []
            if query.text is not None:
                available_modalities.append("text")
            if query.images is not None and len(query.images) > 0:
                available_modalities.append("image")
            if query.audios is not None and len(query.audios) > 0:
                available_modalities.append("audio")
            if query.videos is not None and len(query.videos) > 0:
                available_modalities.append("video")

            # Map tensor outputs to modalities
            if embedding_tensor.dim() == 1:
                primary_modality = (
                    available_modalities[0] if available_modalities else "text"
                )
                modality_embeddings[
                    primary_modality
                ] = embedding_tensor.tolist()
            elif embedding_tensor.dim() == 2:
                for i, modality in enumerate(available_modalities):
                    if i < embedding_tensor.shape[0]:
                        modality_embeddings[modality] = embedding_tensor[
                            i
                        ].tolist()
            else:
                # Handle unexpected tensor dimensions
                if embedding_tensor.dim() > 2:
                    # Flatten to 2D and try again
                    flattened = embedding_tensor.view(
                        embedding_tensor.shape[0], -1
                    )
                    if flattened.shape[0] == len(available_modalities):
                        for i, modality in enumerate(available_modalities):
                            modality_embeddings[modality] = flattened[
                                i
                            ].tolist()
                    else:
                        modality_embeddings[
                            "text"
                        ] = embedding_tensor.flatten().tolist()
                else:
                    # dim() == 0, treat as single text embedding
                    modality_embeddings["text"] = (
                        [embedding_tensor.item()]
                        if embedding_tensor.numel() == 1
                        else embedding_tensor.flatten().tolist()
                    )
        else:
            modality_embeddings["text"] = embedding_tensor.squeeze().tolist()

        return modality_embeddings

    async def generate(self, query: str | Query, context: str) -> str:
        """Generate response to query with context."""
        return self.generator.generate(query=query, context=context)  # type: ignore

    async def batch_generate(
        self, queries: list[str | Query], contexts: list[str]
    ) -> list[str]:
        """Batch generate responses to queries with contexts."""
        if len(queries) != len(contexts):
            raise RAGSystemError(
                "Queries and contexts must have the same length for batch generation."
            )
        return self.generator.generate(query=queries, context=contexts)  # type: ignore

    def _format_context(self, source_nodes: list[SourceNode]) -> str:
        """Format context from nodes retrieved from different modality collections."""
        # Group nodes by modality for better organization
        modality_groups: dict[str, list[SourceNode]] = {}
        for node in source_nodes:
            modality = getattr(node, "modality", "text")
            if modality not in modality_groups:
                modality_groups[modality] = []
            modality_groups[modality].append(node)

        # Modality-specific content extraction rules
        modality_config = {
            "text": {
                "title": "Text Context",
                "content_keys": ["text_content"],
                "prefix": "",
            },
            "image": {
                "title": "Image Context",
                "content_keys": ["text_content", "image_description"],
                "prefix": "Image: ",
            },
            "audio": {
                "title": "Audio Context",
                "content_keys": ["text_content", "audio_transcript"],
                "prefix": "Audio: ",
            },
            "video": {
                "title": "Video Context",
                "content_keys": ["text_content", "video_description"],
                "prefix": "Video: ",
            },
        }

        context_parts = []
        # Process modalities in preferred order
        for modality in ["text", "image", "audio", "video"]:
            if modality in modality_groups:
                config = modality_config[modality]
                descriptions = []

                for node in modality_groups[modality]:
                    content = node.get_content()
                    # Try each content key until we find one
                    for key in config["content_keys"]:
                        if key in content and content[key]:
                            descriptions.append(
                                f"{config['prefix']}{content[key]}"
                            )
                            break

                if descriptions:
                    section = self.rag_config.context_separator.join(
                        descriptions
                    )
                    context_parts.append(f"{config['title']}:\n{section}")

        return "\n\n".join(context_parts)


def _resolve_forward_refs() -> None:
    """Resolve forward references in _RAGSystem."""

    # These imports are needed for Pydantic to resolve forward references
    # ruff: noqa: F401
    from fed_rag.base.generator import BaseGenerator
    from fed_rag.base.knowledge_store import BaseAsyncKnowledgeStore
    from fed_rag.base.retriever import BaseRetriever

    # Update forward references
    _AsyncRAGSystem.model_rebuild()


_resolve_forward_refs()
