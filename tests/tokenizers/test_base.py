from fed_rag.base.tokenizer import BaseTokenizer


def test_generate(dummy_tokenizer: BaseTokenizer) -> None:
    output = dummy_tokenizer.encode("hello world!")
    assert output == [0, 1]
