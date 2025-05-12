# Example: Wikipedia Dec 2021 Knowledge Store

A Docker image providing a pre-built Qdrant vector database with an Atlas corpus knowledge store for retrieval-augmented applications.

⚠️ **Note:** This Docker image is approximately 3.6GB in size due to the included Python environment and ML libraries. Ensure you have sufficient disk space and bandwidth when pulling the image.

## Quick Start

```bash
# Pull the image
docker pull vectorinstitute/qdrant-atlas-dec-wiki-2021:latest

# Run the container with basic settings and gpu acceleration
docker run -d \
  --name qdrant-vector-db \
  --gpus all \
  -p 6333:6333 \
  -p 6334:6334 \
  -v qdrant_data:/qdrant_storage \
  vectorinstitute/qdrant-atlas-dec-wiki-2021:latest
```

## Using the Knowledge Store

Once the container has `healthy` status, then we can use

### Using with fed-rag

```python
from fed_rag.retriever.knowledge_store import QdrantKnowledgeStore
from fed_rag.retrievers.huggingface.hf_sentence_transformer import (
    HFSentenceTransformerRetriever,
)

# build retriever for encoding queries
retriever = HFSentenceTransformerRetriever(
    query_model_name="nthakur/dragon-plus-query-encoder",
    context_model_name="nthakur/dragon-plus-context-encoder",
    load_model_at_init=False,
)

# Connect to the containerized knowledge store
knowledge_store = QdrantKnowledgeStore(
    collection_name="nthakur.dragon-plus-context-encoder",
    host="localhost",
    port=6333,
)

# Retrieve documents
query = "What is the history of marine biology?"
query_emb = retriever.encode_query(query).tolist()

results = knowledge_store.retrieve(query_emb=query_emb, top_k=3)
for node in results:
    print(f"Score: {node.score}, Content: {str(node.node)}")
```

## Acknowledgements

- [Qdrant](https://qdrant.tech/) - Vector Database
- [Facebook AI Research](https://github.com/facebookresearch/atlas) - Atlas Corpus
