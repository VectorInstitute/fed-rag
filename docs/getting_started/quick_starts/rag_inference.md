# Build a RAG System

<!-- markdownlint-disable-file MD033 -->

In this quick start example, we'll demonstrate how to build a RAG system and subequently
query it using FedRAG. We begin with a short primer on the components of RAG
which mirror the abstractions that FedRAG uses to build such systems.

## Components of RAG

<figure markdown="span" id="img_light_mode">
  ![Components of RAG](https://d3ddy8balm3goa.cloudfront.net/vector-ai-pocket-refs/nlp/rag-components.excalidraw.svg)
  <figcaption>The three main components of RAG.</figcaption>
</figure>

<figure markdown="span" id="img_dark_mode">
  ![Components of RAG](https://d3ddy8balm3goa.cloudfront.net/vector-ai-pocket-refs/nlp/rag-components-dark.excalidraw.svg)
  <figcaption>The three main components of RAG.</figcaption>
</figure>

A RAG system is comprised of three main components, namely:

- **Knowledge Store** — contains non-parametric knowledge facts that the system
  can use at inference time in order to produce more accurate responses to queries.
- **Retriever** — a model that takes in a user query and retrieves the most relevant
  knowledge facts from the knowledge store.
- **Generator** — a model that takes in the user's query and additional context
  and provides a response to that query.

!!! note
    The **Retriever** is alos used to populate (i.e index) the **Knowledge Store**
    during setup.

## Building a RAG system

We'll install FedRAG with the `huggingface` extra this time in order to build our
RAG system using HuggingFace models.

``` sh
pip install "fed-rag[huggingface]"
```

### Retriever

``` py title="retriever"
from fed_rag.retrievers.hf_sentence_transformer import (
    HFSentenceTransformerRetriever,
)

QUERY_ENCODER_NAME = "nthakur/dragon-plus-query-encoder"
CONTEXT_ENCODER_NAME = "nthakur/dragon-plus-context-encoder"

retriever = HFSentenceTransformerRetriever(
    query_model_name=QUERY_ENCODER_NAME,
    context_model_name=CONTEXT_ENCODER_NAME,
    load_model_at_init=False,
)
```

### Knowledge Store

``` py title="knowledge artifacts"
import json

# knowledge chunks
chunks_json_strs = [
    '{"id": "0", "title": "Orchid", "text": "Orchids are easily distinguished from other plants, as they share some very evident derived characteristics or synapomorphies. Among these are: bilateral symmetry of the flower (zygomorphism), many resupinate flowers, a nearly always highly modified petal (labellum), fused stamens and carpels, and extremely small seeds"}'
    '{"id": "1", "title": "Tulip", "text": "Tulips are easily distinguished from other plants, as they share some very evident derived characteristics or synapomorphies. Among these are: bilateral symmetry of the flower (zygomorphism), many resupinate flowers, a nearly always highly modified petal (labellum), fused stamens and carpels, and extremely small seeds"}'
]
chunks = [json.loads(line) for line in chunks_json_strs]
```

``` py title="knowledge store"
from fed_rag.knowledge_stores.in_memory import InMemoryKnowledgeStore
from fed_rag.types.knowledge_node import KnowledgeNode, NodeType

knowledge_store = InMemoryKnowledgeStore()

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
```

### Generator

``` py title="generator"
from fed_rag.generators.hf_peft_model import HFPeftModelGenerator
from transformers.generation.utils import GenerationConfig
from transformers.utils.quantization_config import BitsAndBytesConfig

PEFT_MODEL_NAME = "Styxxxx/llama2_7b_lora-quac"
BASE_MODEL_NAME = "meta-llama/Llama-2-7b-hf"

generation_cfg = GenerationConfig(
    do_sample=True,
    eos_token_id=[128000, 128009],
    bos_token_id=128000,
    max_new_tokens=4096,
    top_p=0.9,
    temperature=0.6,
    cache_implementation="offloaded",
    stop_strings="</response>",
)
quantization_config = BitsAndBytesConfig(load_in_4bit=True)
generator = HFPeftModelGenerator(
    model_name=PEFT_MODEL_NAME,
    base_model_name=BASE_MODEL_NAME,
    generation_config=generation_cfg,
    load_model_at_init=False,
    load_model_kwargs={"is_trainable": True, "device_map": "auto"},
    load_base_model_kwargs={
        "device_map": "auto",
        "quantization_config": quantization_config,
    },
)
```

### RAG System

``` py title="RAG system"
from fed_rag.types.rag_system import RAGConfig, RAGSystem

rag_config = RAGConfig(top_k=2)
rag_system = RAGSystem(
    knowledge_store=knowledge_store,
    generator=generator,
    retriever=retriever,
    rag_config=rag_config,
)
```
