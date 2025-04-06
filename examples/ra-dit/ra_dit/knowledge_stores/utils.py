"""Knowledge Store Utils."""

import json
from pathlib import Path

# fed_rag
from fed_rag.knowledge_stores.in_memory import InMemoryKnowledgeStore
from fed_rag.retrievers.hf_sentence_transformer import (
    HFSentenceTransformerRetriever,
)
from fed_rag.types.knowledge_node import KnowledgeNode, NodeType

DATA_DIR = Path(__file__).parents[2].absolute() / "data"
ATLAS_DIR = DATA_DIR / "atlas" / "enwiki-dec2021"


def knowledge_store_from_retriever(
    retriever: HFSentenceTransformerRetriever,
    persist: bool = False,
    data_path: Path | None = None,
) -> InMemoryKnowledgeStore:
    knowledge_store = InMemoryKnowledgeStore()

    # load chunks from atlas corpus
    data_path = data_path if data_path else ATLAS_DIR
    filename = "sm_sample-text-list-100-sec.jsonl"
    with open(data_path / filename) as f:
        chunks = [json.loads(line) for line in f]

    # create knowledge nodes
    nodes = []
    for c in chunks:
        text = c.pop("text")
        node = KnowledgeNode(
            embedding=retriever.encode_context(text).tolist(),
            node_type=NodeType.TEXT,
            text_content=text,
            metadata=c,
        )
        nodes.append(node)

    # load into knowledge_store
    knowledge_store.load_nodes(nodes=nodes)

    # persist
    if persist:
        knowledge_store.persist()

    return knowledge_store
