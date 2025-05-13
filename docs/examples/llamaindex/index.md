# Using LlamaIndex for Inference

## Introduction

After fine-tuning your RAG system to achieve desired performance, you'll want to
deploy it for inference. While FedRAG's `RAGSystem` provides complete inference
capabilities out of the box, you may need additional features for production deployments
or want to leverage the ecosystem of existing RAG frameworks.

FedRAG offers a seamless integration into LlamaIndex[^1]  through our bridges system,
giving you the best of both worlds: FedRAG's fine-tuning capabilities combined
with the extensive inference features of LlamaIndex.

In this example, we demonstrate how you can convert a `RAGSystem` to a
`~llama_index.ManagedIndex` from which you can obtain `~llama_index.QueryEngine`
as well as `~llama_index.Retriever`.

## Installation

To be able to use the LlamaIndex bridge, the `llama-index` extra must be installed.

```sh
pip install fed-rag[llama-index,huggingface]
```

## Setup — The RAG System

```py title="retriever, generator, and knowledge store"
from transformers.generation.utils import GenerationConfig
from fed_rag.generators.huggingface import HFPreTrainedModelGenerator
from fed_rag.retrievers.huggingface.hf_sentence_transformer import (
    HFSentenceTransformerRetriever,
)
from fed_rag.knowledge_stores import InMemoryKnowledgeStore
from fed_rag.types.knowledge_node import KnowledgeNode, NodeType


QUERY_ENCODER_NAME = "nthakur/dragon-plus-query-encoder"
CONTEXT_ENCODER_NAME = "nthakur/dragon-plus-context-encoder"
PRETRAINED_MODEL_NAME = "Qwen/Qwen3-0.6B"

# Retriever
retriever = HFSentenceTransformerRetriever(
    query_model_name=QUERY_ENCODER_NAME,
    context_model_name=CONTEXT_ENCODER_NAME,
    load_model_at_init=False,
)

# Generator
generation_cfg = GenerationConfig(temperature=0.7, max_length=512)
generator = HFPretrainedModelGenerator(
    model_name=PRETRAINED_MODEL_NAME,
    generation_config=generation_cfg,
    load_model_at_init=False,
    load_model_kwargs={"device_map": "auto"},
)

# Knowledge store
knowledge_store = InMemoryKnowledgeStore()

## Add some knowledge
text_chunks = [
    "Retrieval-Augmented Generation (RAG) combines retrieval with generation.",
    "LLMs can hallucinate information when they lack context.",
]
knowledge_nodes = [
    KnowledgeNode(
        node_type="text",
        embedding=retriever.encode_context(ct).tolist(),
        text_content=ct,
    )
    for ct in text_chunks
]
knowledge_store.load_nodes(knowledge_nodes)
```

```py title="assemble the RAGSystem"
from fed_rag.types.rag_system import RAGSystem, RAGConfig

# Create the RAG system
rag_system = RAGSystem(
    retriever=retriever,
    generator=generator,
    knowledge_store=knowledge_store,
    rag_config=RAGConfig(top_k=1),
)
```

## Using the Bridge

Converting your FedRAG system to a LlamaIndex object is seamless since the bridge
functionality is already built into the `RAGSystem` class. The `RAGSystem` inherits
from `LlamaIndexBridgeMixin`, which provides the `to_llamaindex()` method for
effortless conversion.

```py title="Using the LlamaIndex bridge"
# Create a llamaindex object
index = rag_system.to_llamaindex()

# Use it like any other LlamaIndex object to get a query engine
query_engine = index.as_query_engine()
response = query_engine.query(query)
print(response)

# Or, get a retriever
retriever = index.as_retriever()
results = retriever.retrieve(query)
for node in results:
    print(f"Score: {node.score}, Content: {node.node.text}")
```

!!! note
    The `to_llamaindex()` method returns a `FedRAGManagedIndex` object, which is
    a custom implementation of the `~llama_index.BaseManagedIndex` class.

### Modifying Knowledge

In addition to querying the bridged index, you can also make changes to the
underlying KnowledgeStore using LlamaIndex's API:

```py title="Updating the underlying knowledge store"
from llama_index.core.schema import Node, MediaResource

# add nodes
llama_nodes = [
    Node(
        embedding=[1, 1, 1],
        text_resource=MediaResource(text="some arbitrary text"),
    ),
    Node(
        embedding=[2, 2, 2],
        text_resource=MediaResource(text="some more arbitrary text"),
    ),
]
index.insert_nodes(llama_nodes)

# you can also delete nodes
index.delete_nodes(node_ids=[node.node_id for node in llama_nodes])
```

### Advanced Usage

You can combine your bridged index with LlamaIndex's advanced features:

```py title="advanced usage"
from llama_index.core.postprocessor import SimilarityPostprocessor
from llama_index.core.response_synthesizers import TreeSummarize

# Use node postprocessors for filtering retrieved nodes
retriever = index.as_retriever(
    node_postprocessors=[SimilarityPostprocessor(similarity_cutoff=0.7)]
)

# Create a query engine with custom parameters
query_engine = index.as_query_engine(
    response_synthesizer=TreeSummarize(
        verbose=True,
        summary_template="Provide a concise summary of the following: {context}",
    ),
)

# Execute the query with the advanced configuration
response = query_engine.query("Explain the benefits of RAG systems")
print(response)
```

!!! note
    Streaming and async functionalities are not yet supported.

<!-- References -->
[^1]: Liu, J. (2022). LlamaIndex [Computer software]. <https://doi.org/10.5281/zenodo.1234>
