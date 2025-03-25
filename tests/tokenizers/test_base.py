from fed_rag.base.tokenizer import BaseTokenizer


def test_base(mock_tokenizer: BaseTokenizer) -> None:
    input_ids = mock_tokenizer.encode("hello world!")
    decoded_str = mock_tokenizer.decode([1, 2, 3])

    assert input_ids == [0, 1, 2]
    assert decoded_str == "mock decoded sentence"
    assert mock_tokenizer.unwrapped is None
