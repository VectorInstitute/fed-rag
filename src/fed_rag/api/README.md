# fed-rag API

This directory contains the stable, public API for fed-rag. All components
exported from this module follow semantic versioning guarantees.

## Current API Version: 0.1.0 (Pre-1.0 Development)

During pre-1.0 development, minor version increases may include small breaking
changes as the API is refined based on user feedback.

## Core Components

- `RAGSystem`: Main entry point for retrieval-augmented generation
- `RAGConfig`: Configuration options for RAG systems

## Usage Example

```python
from fed_rag.api import RAGSystem, RAGConfig

# Create a RAG system
retriever = ...
knowledge_store = ...
generator = ...

rag = RAGSystem(
    retriever=retriever,
    generator=generator,
    knowledge_store=knowledge_store,
    config=RAGConfig(top_k=5),
)

# Query the system
response = rag.query("How does retrieval-augmented generation work?")
```

## Design

```sh
┌───────────────┐     ┌───────────────┐     ┌───────────────┐
│ Base Classes  │     │  Core         │     │  Components   │
│ & Type Defs   │◄────┤  (RAGSystem)  │◄────┤ (Retrievers,  │
└───────┬───────┘     └───────┬───────┘     │ Generators)   │
        │                     │             └───────┬───────┘
        │                     │                     │
        │                     ▼                     │
        │             ┌───────────────┐             │
        └────────────►│    .api      │◄─────────────┘
                      │ (Public API)  │
                      └───────┬───────┘
                              │
                              ▼
                      ┌───────────────┐
                      │  User Code    │
                      └───────────────┘
```

For project-wide documentation, see the [main README](../../../README.md).
