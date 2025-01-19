# README

FedRAG is a framework for federated fine-tuning of Retrieval-Augmented Generation
(RAG) systems, wherein a server and potentially several client nodes share an overall
model architecture. Since the model is common to all participants of the system,
fine-tuning can be done collaboratively without any of the raw data leaving any
of the client nodes. Instead only the model weight updates are shared between
the client and the server.

```sh
./fed_rag
├── __init__.py
├── base
│   ├── __init__.py
│   ├── loss.py
│   └── models
│       ├── __init__.py
│       ├── generator.py # BaseGeneratorModel
│       ├── rag.py # BaseRAGModel
│       └── retriever.py # BaseRetrieverModel
├── loss
│   ├── __init__.py
│   ├── generator  # Losses for generation task
│   │   └── __init__.py
│   └── retriever # Losses for retrieval task
│       └── __init__.py
├── models
│   ├── __init__.py
│   ├── generators # Generator models
│   │   └── __init__.py
│   └── retrievers # Retrieval models
│       └── __init__.py
├── ops # module for running fed system
│   └── __init__.py
└── types
    ├── __init__.py
    ├── client.py # Wrapper for flwr.Client
    └── server.py # Wrapper for flwr.Server
```

## Getting Started

## Contributing

## Examples
