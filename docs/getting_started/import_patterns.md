# Import Patterns

FedRAG provides a carefully designed public API for working with RAG and both centralized
and federated fine-tuning components. All components exported at the root level
and from public subpackages are considered stable and follow semantic versioning guidelines.

## Root Imports

Import core components directly from the root:

```py
from fed_rag import (
    RAGSystem,
    RAGConfig,
    HFPretrainedModelGenerator,
    HFSentenceTransformerRetriever,
    InMemoryKnowledgeStore,
)

# Now use the components directly
system = RAGSystem(
    retriever=HFSentenceTransformerRetriever(...),
    generator=HFPretrainedModelGenerator(...),
    knowledge_store=InMemoryKnowledgeStore(),
    rag_config=RAGConfig(...),
)
```

## Namespaced Imports

For better organization and increased clarity, you can import from specific
component categories:

```py
from fed_rag.core import RAGSystem
from fed_rag.data_structures.rag import RAGConfig
from fed_rag.generators import HFPretrainedModelGenerator
from fed_rag.retrievers import HFSentenceTransformerRetriever
from fed_rag.knowledge_stores import InMemoryKnowledgeStore

# Create system with components from different namespaces
system = RAGSystem(
    retriever=HFSentenceTransformerRetriever(...),
    generator=HFPretrainedModelGenerator(...),
    knowledge_store=InMemoryKnowledgeStore(),
    rag_config=RAGConfig(...),
)
```

!!! note
    Modules and functions prefixed with an underscore (e.g., `_internal`) are considered
    implementation details and may change between versions.
