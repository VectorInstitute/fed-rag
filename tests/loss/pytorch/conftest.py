import pytest
import torch

EMB_DIM = 10
BATCH_SIZE = 2


@pytest.fixture()
def retrieved_chunks() -> torch.Tensor:
    """Embeddings of 'retrieved' chunks."""
    num_chunks = 3

    batch = []
    for bx in range(1, BATCH_SIZE + 1):
        embs = []
        for ix in range(1, num_chunks + 1):
            embs.append([bx / ix for _ in range(EMB_DIM)])
        batch.append(embs)

    return torch.tensor(batch, dtype=torch.float32)


@pytest.fixture()
def context() -> torch.Tensor:
    batch = []
    for ix in range(1, BATCH_SIZE):
        batch.append(torch.ones(EMB_DIM) * ix)
    return torch.stack(batch, dim=0)
