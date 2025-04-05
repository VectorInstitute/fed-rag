# Standard Usage

The standard usage pattern for FedRAG aligns to a natural RAG fine-tuning workflow,
and looks as follows:

1. Build a [`RAGSystem`](../api_reference/rag_system/index.md)
2. Create a [`RAGFinetuningDataset`](../api_reference/finetuning_datasets/index.md)
3. Define a training loop and evaluation function and decorate both of these with
the appropriate [`decorators`](../api_reference/decorators/index.md).
4. Create an [`FLTask`](../api_reference/fl_tasks/index.md)
5. Spin up FL servers and FL clients to begin federated fine-tuning!

!!! note
    Steps 1. through 3. are—minus the decoration of trainers and testers—typical
    steps one would perform for a centralized RAG fine-tuning task.

!!! tip
    Before proceeding to federated learning, one should verify that the centralized
    task runs as intended with a representative dataset. In fact, centralized learning
    represents a standard baseline with which to compare federated learning results.
