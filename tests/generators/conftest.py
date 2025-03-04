import pytest

from fed_rag.base.generator import BaseGenerator


class MockGenerator(BaseGenerator):
    def generate(self, input: str) -> str:
        return f"mock output from '{input}'."


@pytest.fixture
def mock_generator() -> BaseGenerator:
    return MockGenerator()
