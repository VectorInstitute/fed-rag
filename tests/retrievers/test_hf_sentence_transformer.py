from unittest.mock import MagicMock, patch

from fed_rag.base.retriever import BaseRetriever
from fed_rag.retrievers.hf_sentence_transformer import (
    HFSentenceTransformerRetriever,
)


def test_hf_pretrained_generator_class() -> None:
    names_of_base_classes = [
        b.__name__ for b in HFSentenceTransformerRetriever.__mro__
    ]
    assert BaseRetriever.__name__ in names_of_base_classes


@patch.object(HFSentenceTransformerRetriever, "_load_model_from_hf")
def test_hf_pretrained_generator_class_init_no_load(
    mock_load_from_hf: MagicMock,
) -> None:
    retriever = HFSentenceTransformerRetriever(
        model_name="fake_name", load_model_at_init=False
    )

    assert retriever.model_name == "fake_name"
    assert retriever._encoder is None
    assert retriever._query_encoder is None
    assert retriever._context_encoder is None
