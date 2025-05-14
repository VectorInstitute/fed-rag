# Integrations

FedRAG offers integrations with popular frameworks and tools across the RAG ecosystem.
This page documents currently supported integrations and our roadmap for future compatibility.

!!! info "Status Legend"
    :material-check-bold: — Currently supported;
    :material-clock: — Planned (linked to GitHub issue);
    Empty — Not currently planned

## Deep learning libraries

| Framework  |         Status        |
|------------|:---------------------:|
| PyTorch    | :material-check-bold: |
| Keras      |                       |
| TensorFlow |                       |
| Jax        |                       |

## Fine-tuning frameworks

| Framework   |                                  Status                                 |
|-------------|:-----------------------------------------------------------------------:|
| HuggingFace | :material-check-bold:                                                    |
| Unsloth     | [:material-clock:](https://github.com/VectorInstitute/fed-rag/issues/94) |

## RAG inference frameworks

| Framework  |                                  Status                                   |
|------------|:-------------------------------------------------------------------------:|
| LlamaIndex | :material-check-bold:                                                     |
| LangChain  | [:material-clock:](https://github.com/VectorInstitute/fed-rag/issues/291) |
| Haystack   |                                                                           |

## Knowledge Stores

| Storage Solution |                                  Status                                   |
|------------------|:-------------------------------------------------------------------------:|
| Qdrant           | :material-check-bold:                                                     |
| ChromaDB         | [:material-clock:](https://github.com/VectorInstitute/fed-rag/issues/293) |
| FAISS            | [:material-clock:](https://github.com/VectorInstitute/fed-rag/issues/292) |
| PGVector         |                                                                           |

!!! note "Contributing Integrations"
    We welcome community contributions for additional integrations. See our
    [contribution guidelines](./CONTRIBUTING.md) for more information on
    implementing and submitting new integrations.
