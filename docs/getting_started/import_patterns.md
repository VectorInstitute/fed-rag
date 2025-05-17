# Import Patterns

FedRAG offers multiple import patterns to suit different coding styles and preferences.
All patterns use the stable public API, so you can choose the style that works best for your project.

!!! note
    The stable public API is versioned separately from the main library. Currently,
    this API is pre v1, and while we aim for stability, components here may change
    during the pre v1 development phase.

!!! info
    You may still import from the internal API i.e, `fed_rag.types` or
    `fed_rag.trainers` and etc., but it is in the intention that in the future
    that for most use cases, imports should come from the public API: `fed_rag.api`.

## Flat Imports from API

Import everything you need directly from the main API module:

```py
from fed_rag.api import (
    RAGSystem,
    HFPretrainedModelGenerator,
    HFSentenceTransformerRetriever,
    InMemoryKnowledgeStore,
)

# Now use the components directly
system = RAGSystem(
    retriever=HFSentenceTransformerRetriever(...),
    generator=HFPretrainedModelGenerator(...),
    knowledge_store=InMemoryKnowledgeStore(),
)
```

## Namespaced Imports

For better organization and increased clarity, you can import from specific
component categories:

```py
from fed_rag.api.core import RAGSystem
from fed_rag.api.types import RAGConfig
from fed_rag.api.generators import HFPretrainedModelGenerator
from fed_rag.api.retrievers import HFSentenceTransformerRetriever
from fed_rag.api.knowledge_stores import InMemoryKnowledgeStore

# Create system with components from different namespaces
system = RAGSystem(
    retriever=HFSentenceTransformerRetriever(...),
    generator=HFPretrainedModelGenerator(...),
    knowledge_store=InMemoryKnowledgeStore(),
    rag_config=RAGConfig(...),
)
```
