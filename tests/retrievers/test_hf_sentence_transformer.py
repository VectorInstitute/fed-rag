from unittest.mock import MagicMock, patch

from sentence_transformers import SentenceTransformer

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
    dummy_sentence_transformer: SentenceTransformer,
) -> None:
    retriever = HFSentenceTransformerRetriever(
        model_name="fake_name", load_model_at_init=False
    )

    assert retriever.model_name == "fake_name"
    assert retriever._encoder is None
    assert retriever._query_encoder is None
    assert retriever._context_encoder is None

    # load model
    mock_load_from_hf.return_value = dummy_sentence_transformer
    retriever._load_model_from_hf(load_type="encoder")
    args, kwargs = mock_load_from_hf.call_args

    # assert
    mock_load_from_hf.assert_called_once()
    assert retriever.encoder == dummy_sentence_transformer
    assert args == ()
    assert kwargs == {"load_type": "encoder"}
