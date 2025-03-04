from fed_rag.base.generator import BaseGenerator


def test_generate(mock_generator: BaseGenerator) -> None:
    output = mock_generator.generate("hello")
    assert output == "mock output from 'hello'."
