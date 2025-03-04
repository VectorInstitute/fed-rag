import pytest
import torch

from fed_rag.base.generator import BaseGenerator


class MockGenerator(BaseGenerator):
    def generate(self, input: str) -> str:
        return f"mock output from '{input}'."

    @property
    def model(self) -> torch.nn.Module:
        return torch.nn.Linear(2, 1)


@pytest.fixture
def mock_generator() -> BaseGenerator:
    return MockGenerator()
