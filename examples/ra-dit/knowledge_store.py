"""Knowledge Store."""

from fed_rag.knowledge_stores.in_memory import InMemoryKnowledgeStore
from fed_rag.retrievers.hf_sentence_transformer import (
    HFSentenceTransformerRetriever,
)

knowledge_store = InMemoryKnowledgeStore()
dragon_retriever = HFSentenceTransformerRetriever(
    query_model_name="nthakur/dragon-plus-query-encoder",
    context_model_name="nthakur/dragon-plus-context-encoder",
)

# load chunks from atlas corpus into knowledge store

# persist to disk
