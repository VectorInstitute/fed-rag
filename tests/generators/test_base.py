from fed_rag.base.generator import BaseGenerator


def test_generate(mock_generator: BaseGenerator) -> None:
    output = mock_generator.generate(query="hello", context="again")
    assert output == "mock output from 'hello' and 'again'."


def test_compute_target_sequence_proba(mock_generator: BaseGenerator) -> None:
    proba = mock_generator.compute_target_sequence_proba(
        prompt="mock prompt", target="mock target"
    )
    assert proba == 0.42
