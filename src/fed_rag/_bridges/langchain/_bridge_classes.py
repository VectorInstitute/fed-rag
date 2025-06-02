import uuid
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any, Callable, Generator, Iterable

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseLLM
from langchain_core.outputs import Generation, LLMResult
from langchain_core.vectorstores import VectorStore

from fed_rag.base.generator import BaseGenerator
from fed_rag.core.rag_system._synchronous import _RAGSystem
from fed_rag.data_structures import KnowledgeNode
from fed_rag.exceptions import BridgeError

if TYPE_CHECKING:  # pragma: no cover
    from torch import Tensor


class FedRAGVectorStore(VectorStore):
    """A ~langchain_core.vectorstores.VectorStore adapter for fed_rag._RAGSystem.

    Can be converted to a ~langchain_core.vectorstores.VectorStoreRetriever
    using the `as_retriever` method.
    """

    class FedRAGEmbeddings(Embeddings):
        def __init__(
            self, encode_query: Callable[[str | list[str]], "Tensor"]
        ):
            self.encode_query = encode_query

        def embed_documents(self, texts: list[str]) -> list[list[float]]:
            return self.encode_query(texts).tolist()  # type: ignore

        def embed_query(self, text: str) -> list[float]:
            return self.encode_query(text).tolist()  # type: ignore

    def __init__(self, rag_system: "_RAGSystem"):
        """Initialize the FedRAG Vector Store."""
        super().__init__()
        self._rag_system = rag_system

    @property
    def embeddings(self) -> Embeddings:
        """Return the embeddings used by the vector store."""
        return self.FedRAGEmbeddings(
            encode_query=self._rag_system.retriever.encode_query
        )

    def delete(self, ids: list[str] | None = None, **kwargs: Any) -> bool:
        """Delete nodes from the vector store.

        Args:
            ids (list[str], optional): List of node IDs to delete. If None, no nodes are deleted.
            **kwargs (Any): Additional keyword arguments (not used).

        Returns:
            bool: True if nodes were deleted, False if no IDs were provided.
        """
        if ids is None:
            return False
        for id_ in ids:
            self._rag_system.knowledge_store.delete_node(id_)
        return True

    def add_texts(
        self,
        texts: Iterable[str],
        metadatas: list[dict[str, Any]] | None = None,
        ids: list[str] | None = None,
        **kwargs: Any,
    ) -> list[str]:
        """Add texts to the vector store.

        Args:
            texts (Iterable[str]): Texts to add.
            metadatas (list[dict[str, Any]], optional): List of metadata dictionaries for each text. Defaults to None.
            ids (list[str], optional): List of IDs for the nodes. If None, IDs will be generated automatically.
            **kwargs (Any): Additional keyword arguments (not used).

        Returns:
            list[str]: List of node IDs for the added texts.
        """
        texts = list(texts)
        if not ids:
            ids = [str(uuid.uuid4()) for _ in texts]
        if not metadatas:
            metadatas = [{} for _ in texts]
        if len(texts) != len(metadatas) or len(texts) != len(ids):
            raise ValueError(
                "The number of texts, metadatas, and ids must match."
            )
        nodes = [
            KnowledgeNode(
                node_id=id_,
                embedding=self.embeddings.embed_query(text),
                node_type="text",
                text_content=text,
                metadata=metadata,
            )
            for text, metadata, id_ in zip(texts, metadatas, ids)
        ]
        self._rag_system.knowledge_store.load_nodes(nodes)
        return ids

    def add_documents(
        self,
        documents: list[Document],
        ids: list[str] | None = None,
        **kwargs: Any,
    ) -> list[str]:
        """Add documents to the vector store.

        Args:
            documents (list[Document]): List of documents to add.
            ids (list[str], optional): List of IDs for the nodes. If None, IDs will be generated automatically.
            **kwargs (Any): Additional keyword arguments (not used).

        Returns:
            list[str]: List of node IDs for the added documents.
        """
        if not ids:
            # Documents may have ids, see ~langchain_core.documents.base.BaseMedia
            ids = [doc.id or str(uuid.uuid4()) for doc in documents]
        nodes = [
            KnowledgeNode(
                node_id=id_,
                embedding=self.embeddings.embed_query(doc.page_content),
                node_type="text",
                text_content=doc.page_content,
                metadata=doc.metadata,
            )
            for doc, id_ in zip(documents, ids)
        ]
        self._rag_system.knowledge_store.load_nodes(nodes)
        return ids

    def similarity_search(
        self, query: str, k: int = 4, **kwargs: Any
    ) -> list[Document]:
        """Perform similarity search.

        Args:
            query (str): The query string to search against.
            k (int): The number of top results to return. Defaults to 4.
            **kwargs (Any): Additional keyword arguments (not used).

        Returns:
            list[Document]: A list of Documents that are similar to the query.
        """
        embedding = self.embeddings.embed_query(query)
        return self.similarity_search_by_vector(embedding, k, **kwargs)

    def similarity_search_by_vector(
        self, embedding: list[float], k: int = 4, **kwargs: Any
    ) -> list[Document]:
        """Perform similarity search by vector.

        Args:
            embedding (list[float]): The embedding vector to search against.
            k (int): The number of top results to return. Defaults to 4.
            **kwargs (Any): Additional keyword arguments (not used).

        Returns:
            list[Document]: A list of Documents that are similar to the embedding.
        """
        return [
            doc
            for doc, _ in self.similarity_search_with_score_by_vector(
                embedding, k, **kwargs
            )
        ]

    def similarity_search_with_score(
        self, query: str, k: int = 4, **kwargs: Any
    ) -> list[tuple[Document, float]]:
        """Perform similarity search with score.

        Args:
            query (str): The query string to search against.
            k (int): The number of top results to return. Defaults to 4.
            **kwargs (Any): Additional keyword arguments (not used).

        Returns:
            list[tuple[Document, float]]: A list of tuples containing the Document and its score.
        """
        embedding = self.embeddings.embed_query(query)
        return self.similarity_search_with_score_by_vector(
            embedding, k, **kwargs
        )

    def similarity_search_with_score_by_vector(
        self,
        embedding: list[float],
        k: int = 4,
        **kwargs: Any,
    ) -> list[tuple[Document, float]]:
        """Perform similarity search with score by vector.

        Args:
            embedding (list[float]): The embedding vector to search against.
            k (int): The number of top results to return. Defaults to 4.
            **kwargs (Any): Additional keyword arguments (not used).

        Returns:
            list[tuple[Document, float]]: A list of tuples containing the Document and its score.
        """
        return [
            (
                Document(
                    id=node.node_id,
                    page_content=node.text_content or "",
                    metadata=node.metadata,
                ),
                score,
            )
            for score, node in self._rag_system.knowledge_store.retrieve(
                query_emb=embedding, top_k=k
            )
        ]

    def _select_relevance_score_fn(self) -> Callable[[float], float]:
        """Select the relevance score function for the vector store.

        NOTE: Only cosine similarity is supported for now.
        """
        return self._cosine_relevance_score_fn  # type: ignore

    @classmethod
    def from_texts(cls: type[VectorStore], **kwargs: Any) -> type[VectorStore]:
        raise BridgeError(
            "FedRAGVectorStore does not support from_texts method. "
            "Use the add_texts method to add texts to the vector store."
        )


class FedRAGLLM(BaseLLM):
    def __init__(self, rag_system: "_RAGSystem"):
        """Initialize the FedRAG LLM."""
        super().__init__()
        self._rag_system = rag_system

    @property
    def _llm_type(self) -> str:
        """Return the type of the LLM."""
        return "fed_rag.generator"

    @contextmanager
    def _generator(
        self,
    ) -> Generator[BaseGenerator, None, None]:
        original_template = self._rag_system.generator.prompt_template
        try:
            template_for_llamaindex = "{query}"
            self._rag_system.generator.prompt_template = (
                template_for_llamaindex
            )

            yield self._rag_system.generator
        finally:
            self._rag_system.generator.prompt_template = original_template

    def _generate(
        self,
        prompts: list[str],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> LLMResult:
        """Generate text using the FedRAG system.

        Args:
            prompts (list[str]): List of prompts to generate text for.
            stop (list[str], optional): List of stop sequences. Not supported in this implementation.
            **kwargs (Any): Additional keyword arguments (not used).

        Returns:
            LLMResult: The result of the text generation containing generations.
        """
        if stop is not None:
            raise BridgeError(
                "FedRAGLLM does not support stop sequences. "
                "Please use the generator directly if you need this feature."
            )
        with self._generator() as generator:
            generations = [
                [Generation(text=generator.generate(query=prompt, context=""))]
                for prompt in prompts
            ]
        return LLMResult(generations=generations)

    def _stream(
        self,
        prompt: str,
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> Iterable[Generation]:
        """Stream text generation.

        NOTE: This method is not supported in FedRAGLLM.
        """
        raise BridgeError(
            "FedRAGLLM does not support streaming. "
            "Please use the generator directly if you need this feature."
        )
