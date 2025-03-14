"""Knowledge Store."""

import json
from pathlib import Path

from fed_rag.knowledge_stores.in_memory import InMemoryKnowledgeStore
from fed_rag.retrievers.hf_sentence_transformer import (
    HFSentenceTransformerRetriever,
)
from fed_rag.types.knowledge_node import KnowledgeNode, NodeType

DATA_PATH = Path(__file__).parents[1].absolute() / "data"

knowledge_store = InMemoryKnowledgeStore()
dragon_retriever = HFSentenceTransformerRetriever(
    query_model_name="nthakur/dragon-plus-query-encoder",
    context_model_name="nthakur/dragon-plus-context-encoder",
)

# load chunks from atlas corpus
filename = "mock_data.jsonl"
with open(DATA_PATH / filename) as f:
    chunks = [json.loads(line) for line in f]

# create knowledge nodes
nodes = []
for c in chunks:
    node = KnowledgeNode(
        embedding=dragon_retriever.encode_context(c["text"]).tolist(),
        node_type=NodeType.TEXT,
        text_content=c["text"],
        metadata={"title": c["title"], "id": c["id"]},
    )
    nodes.append(node)

# load into knowledge_store
knowledge_store.load_nodes(nodes=nodes)

# persist to disk

if __name__ == "__main__":
    print(knowledge_store.count)
