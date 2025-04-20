import math

import pytest
import torch

EMB_DIM = 10
BATCH_SIZE = 2
NUM_CHUNKS = 3


@pytest.fixture()
def retrieved_chunks() -> torch.Tensor:
    """Embeddings of 'retrieved' chunks."""
    batch = []
    for bx in range(1, BATCH_SIZE + 1):
        embs = []
        for ix in range(1, NUM_CHUNKS + 1):
            embs.append([bx / ix for _ in range(EMB_DIM)])
        batch.append(embs)

    return torch.tensor(batch, dtype=torch.float32)


@pytest.fixture()
def contexts() -> torch.Tensor:
    batch = []
    for ix in range(1, BATCH_SIZE):
        batch.append(torch.ones(EMB_DIM) * ix)
    return torch.stack(batch, dim=0)


@pytest.fixture()
def lm_scores() -> torch.Tensor:
    """Mock probas of generated outputs 'given' context and chunk."""
    batch = []
    for bx in range(1, BATCH_SIZE + 1):
        scores = [math.exp(ix) for ix in range(NUM_CHUNKS)]
        scores = [el / sum(scores) for el in scores]
        batch.append(scores)

    return torch.tensor(batch, dtype=torch.float32)
