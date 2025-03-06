from unittest.mock import MagicMock, _Call, patch

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
def test_hf_pretrained_generator_class_init_delayed_load(
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


@patch.object(HFSentenceTransformerRetriever, "_load_model_from_hf")
def test_hf_pretrained_generator_class_init_delayed_dual_encoder_load(
    mock_load_from_hf: MagicMock,
    dummy_sentence_transformer: SentenceTransformer,
) -> None:
    retriever = HFSentenceTransformerRetriever(
        query_model_name="query_fake_name",
        context_model_name="context_fake_name",
        load_model_at_init=False,
    )

    assert retriever.model_name is None
    assert retriever.query_model_name == "query_fake_name"
    assert retriever.context_model_name == "context_fake_name"
    assert retriever._encoder is None
    assert retriever._query_encoder is None
    assert retriever._context_encoder is None

    # load models
    mock_load_from_hf.return_value = dummy_sentence_transformer
    retriever._load_model_from_hf(load_type="query_encoder")
    retriever._load_model_from_hf(load_type="context_encoder")

    # assert
    calls = [
        _Call(((), {"load_type": "query_encoder"})),
        _Call(((), {"load_type": "context_encoder"})),
    ]
    mock_load_from_hf.assert_has_calls(calls)
    assert retriever.query_encoder == dummy_sentence_transformer
    assert retriever.context_encoder == dummy_sentence_transformer
