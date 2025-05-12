# Build a Qdrant Knowledge Store

## Introduction

For their RAG system, the authors of the RA-DIT[^1] paper, used a knowledge
store that was comprised of artifacts from two sources:

1. Text chunks from the Dec. 20, 2021 Wikipedia dump. [^2]
2. A sample of text chunks from the 2017-2020 CommonCrawl dumps. [^3]

In total, their knowledge store (which they referred to as a "retrieval corpus")
consisted of 399M knowledge nodes, each having text content with no more than
200 words.

## Our Scaled Down Knowledge Store

### The raw text chunks

For practical purposes, in this example, we only consider the text chunks from
the Dec. 20, 2021 Wikipedia dump. Moreover, we only make use of the file
`text-list-100-sec.jsonl` from this dump, which contains 33,176,581 chunks that
follow the JSON schema depicted below.

```json title="A text chunk"
chunk = {
    "id": "...",
    "title": "...",
    "section": "...",
    "text": "..."
}
```

### Creating a `KnowledgeNode`

Each raw text chunk is converted to a [`KnowledgeNode`](../../api_reference/knowledge_nodes/index.md),
using a simple template for preparing the node's `text_content`.

```py title="Creating a KnowledgeNode code snippet"
import json
from fed_rag.types.knowledge_node import KnowledgeNode

chunk = json.loads(chunk_json_str)
context_text = (
    f"title: {chunk.pop('title')]}"  # (1)!
    f"\nsection: {chunk.pop('section')}"
    f"\ntext: {chunk.pop('text')}"
)
embedding = retriever.encode_context(context_text).tolist()  # (2)!

node = KnowledgeNode(
    embedding=embedding,
    node_type=NodeType.TEXT,
    text_content=context_text,
    metadata=chunk,
)
```

1. A simple template for creating the node's context from the chunk's data.
2. A `HuggingFaceSentenceTransformerRetriever` was built using the models:
`nthakur/dragon-plus-context-encoder` and `nthakur/dragon-plus-query-encoder`.

### Adding nodes to a `QdrantKnowledgeStore`

In this example, we use the Qdrant extra and add our knowledge nodes to a locally
running `QdrantKnowledgeStore`. The code snippet below shows how to instantiate
such a knowledge store and subsequently add a node to it.

In the accompanying code, we provide users the options to change their retriever
`~sentence_transformer.SentenceTransformer` model, and also change the number of
raw text chunks to index.

```py title="Adding our nodes to a QdrantKnowledgeStore"
from fed_rag.knowledge_stores.qdrant import QdrantKnowledgeStore

knowledge_store = QdrantKnowledgeStore(
    collection_name="nthakur.dragon-plus-context-encoder"
)

# add node to knowledge store
knowledge_store.load_node(node)
```

## Getting The Knowledge Store

For convenience, we provide a Docker image that contains a pre-built Qdrant
vector database that builds the knowledge store as described above.

To get this image, you can pull it from Vector Institute's docker hub:

```sh
# Pull the image
docker pull vectorinstitute/qdrant-atlas-dec-wiki-2021:latest
```

!!! note
    This Docker image is approximately 3.6GB in size due to the included Python
    environment and ML libraries. Ensure you have sufficient disk space and
    bandwidth when pulling the image.

The image can then be executed with the command provided below. This code
will download the raw Wikipedia text chunks, encode them using the default
retriever model (i.e., `nthakur/dragon-plus-context-encoder`) and add them to
the Qdrant vector database.

```sh title="Running the docker image"
# Run the container with basic settings and gpu acceleration
docker run -d \
  --name qdrant-vector-db \
  --gpus all \
  -p 6333:6333 \
  -p 6334:6334 \  # needed for gRPC
  -v qdrant_data:/qdrant_storage \
  vectorinstitute/qdrant-atlas-dec-wiki-2021:latest
```

The above command runs in detached mode and will create a container called `qdrant-vector-db`.
Monitor the logs of the container to determine the progress. Once the container is
`healthy` then it can be used.

The command above launches a detached container named `qdrant-vector-db`. You can
monitor its progress through the container logs via:

```sh
docker logs qdrant-vector-db
```

The knowledge store is ready for use once the container status shows as `healthy`.

!!! note
    Building the knowledge store with all 33,176,581 text chunks can take 4-7 days,
    depending on your hardware setup. While our code provides a solid foundation,
    you can implement further optimizations based on your specific performanc
    requirements and infrastructure.

!!! tip
    To quickly verify the Docker image works correctly, use the parameter `-e SAMPLE_SIZE=tiny`
    when running the container. This executes the process on a small subset of
    Wikipedia text chunks, allowing for rapid validation before committing to
    a larger subset or the full dataset.

### Testing the knowledge store

Once the container shows `healthy`, the knowledge store can be used. Below is a
code snippet for quickly testing that nodes can be successfully retrieved from it.

```py title="Testing the knowledge store with FedRAG"
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
)

# Retrieve documents
query = "What is the history of marine biology?"
query_emb = retriever.encode_query(query).tolist()

results = knowledge_store.retrieve(query_emb=query_emb, top_k=3)
for node in results:
    print(f"Score: {node.score}, Content: {str(node.node)}")
```

### More details about the Docker image

For comprehensive information about the prepared Docker image and its available
configuration options, visit: <https://github.com/VectorInstitute/fed-rag/tree/main/examples/knowledge_stores/ra-dit-ks>.

## What's Next?

Now that our knowledge store is complete, we'll proceed to construct the RAG system
for fine-tuning.

All code related to the knowledge store implementation can be found in the
`examples/ra-dit/knowledge_store/` directory.

<!-- References -->
[^1]: Lin, Xi Victoria, et al. "Ra-dit: Retrieval-augmented dual instruction tuning."
  The Twelfth International Conference on Learning Representations. 2023.
[^2]: Common Crawl Foundation. (2017-2020). Common Crawl. Retrieved from <https://commoncrawl.org/>
[^3]: Izacard, Gautier, et al. "Few-shot learning with retrieval augmented language
  models." arXiv preprint arXiv:2208.03299 1.2 (2022): 4.
