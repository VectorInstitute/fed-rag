import pytest
import torch


@pytest.fixture()
def input_and_target_ids() -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    input_ids = [torch.zeros(3)] * 3
    target_ids = [torch.ones(3)] * 3
    return input_ids, target_ids
