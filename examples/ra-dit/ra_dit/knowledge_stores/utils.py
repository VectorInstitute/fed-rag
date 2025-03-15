"""Knowledge Store Utils."""

import json
from pathlib import Path

# fed_rag
from fed_rag.knowledge_stores.in_memory import InMemoryKnowledgeStore
from fed_rag.retrievers.hf_sentence_transformer import (
    HFSentenceTransformerRetriever,
)
from fed_rag.types.knowledge_node import KnowledgeNode, NodeType

DATA_PATH = Path(__file__).parents[2].absolute() / "data"


def knowledge_store_from_retriever(
    retriever: HFSentenceTransformerRetriever,
    persist: bool = False,
    out_path: Path | None = None,
) -> InMemoryKnowledgeStore:
    knowledge_store = InMemoryKnowledgeStore()

    # load chunks from atlas corpus
    filename = "mock_data.jsonl"
    with open(DATA_PATH / filename) as f:
        chunks = [json.loads(line) for line in f]

    # create knowledge nodes
    nodes = []
    for c in chunks:
        node = KnowledgeNode(
            embedding=retriever.encode_context(c["text"]).tolist(),
            node_type=NodeType.TEXT,
            text_content=c["text"],
            metadata={"title": c["title"], "id": c["id"]},
        )
        nodes.append(node)

    # load into knowledge_store
    knowledge_store.load_nodes(nodes=nodes)

    # persist
    # if persist:
    #     if out_path is None:
    #         raise ValueError(
    #             "`out_path` is `None` but `persist` is `True`. Must provide a valid `out_path`."
    #         )
    #     knowledge_store.persist(out_path)

    return knowledge_store
