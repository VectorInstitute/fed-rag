from unittest.mock import MagicMock, _Call, patch

import pytest
from sentence_transformers import SentenceTransformer

from fed_rag.base.retriever import BaseRetriever
from fed_rag.retrievers.hf_sentence_transformer import (
    HFSentenceTransformerRetriever,
    InvalidLoadType,
    LoadKwargs,
)


def test_hf_sentence_transformer_retriever_class() -> None:
    names_of_base_classes = [
        b.__name__ for b in HFSentenceTransformerRetriever.__mro__
    ]
    assert BaseRetriever.__name__ in names_of_base_classes


@patch.object(HFSentenceTransformerRetriever, "_load_model_from_hf")
def test_hf_sentence_transformer_retriever_class_init_delayed_load(
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
def test_hf_sentence_transformer_retriever_class_init_delayed_dual_encoder_load(
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


@patch.object(HFSentenceTransformerRetriever, "_load_model_from_hf")
def test_hf_sentence_transformer_retriever_class_init_dual_encoder(
    mock_load_from_hf: MagicMock,
    dummy_sentence_transformer: SentenceTransformer,
) -> None:
    # arrange
    mock_load_from_hf.return_value = dummy_sentence_transformer

    # act
    retriever = HFSentenceTransformerRetriever(
        query_model_name="query_fake_name",
        context_model_name="context_fake_name",
    )

    # assert
    calls = [
        _Call(((), {"load_type": "query_encoder"})),
        _Call(((), {"load_type": "context_encoder"})),
    ]
    mock_load_from_hf.assert_has_calls(calls)
    assert retriever.encoder is None
    assert retriever.query_encoder == dummy_sentence_transformer
    assert retriever.context_encoder == dummy_sentence_transformer


@patch.object(HFSentenceTransformerRetriever, "_load_model_from_hf")
def test_hf_sentence_transformer_retriever_class_init_encoder(
    mock_load_from_hf: MagicMock,
    dummy_sentence_transformer: SentenceTransformer,
) -> None:
    # arrange
    mock_load_from_hf.return_value = dummy_sentence_transformer

    # act
    retriever = HFSentenceTransformerRetriever(
        model_name="fake_name",
    )

    # assert
    calls = [
        _Call(((), {"load_type": "encoder"})),
    ]
    mock_load_from_hf.assert_has_calls(calls)
    assert retriever.encoder == dummy_sentence_transformer
    assert retriever.query_encoder is None
    assert retriever.context_encoder is None


@patch("fed_rag.retrievers.hf_sentence_transformer.SentenceTransformer")
def test_load_model_from_hf_constructs_sentence_transformer_obj(
    mock_sentence_transformer: MagicMock,
) -> None:
    # arrange
    # act
    retriever = HFSentenceTransformerRetriever(
        model_name="fake_name", load_model_kwargs={"device": "cpu"}
    )

    # assert
    mock_sentence_transformer.assert_called_once_with(
        "fake_name", device="cpu"
    )
    assert retriever.query_encoder is None
    assert retriever.context_encoder is None


@patch("fed_rag.retrievers.hf_sentence_transformer.SentenceTransformer")
def test_load_model_from_hf_constructs_sentence_transformer_obj_dual(
    mock_sentence_transformer: MagicMock,
) -> None:
    # arrange
    # act
    retriever = HFSentenceTransformerRetriever(
        query_model_name="fake_query_model_name",
        context_model_name="fake_context_model_name",
        load_model_kwargs=LoadKwargs(
            query_encoder={"device": "cuda:0"},
            context_encoder={"device": "cpu"},
        ),
    )

    # assert
    mock_sentence_transformer.assert_has_calls(
        [
            _Call((("fake_query_model_name",), {"device": "cuda:0"})),
            _Call((("fake_context_model_name",), {"device": "cpu"})),
        ]
    )
    assert retriever.encoder is None


def test_load_model_from_hf_raises_invalid_type_error() -> None:
    # arrange
    retriever = HFSentenceTransformerRetriever(
        model_name="fake_name", load_model_at_init=False
    )

    # act/assert
    with pytest.raises(InvalidLoadType, match="Invalid `load_type` supplied."):
        retriever._load_model_from_hf(load_type="unsupported_type")


@patch.object(HFSentenceTransformerRetriever, "_load_model_from_hf")
def test_encode_query(
    mock_load_from_hf: MagicMock,
) -> None:
    # arrange
    mock_encoder = MagicMock()
    mock_encoder.encode.side_effect = iter([[1, 2, 3], [4, 5, 6]])
    mock_load_from_hf.return_value = mock_encoder
    retriever = HFSentenceTransformerRetriever(
        model_name="query_fake_name",
    )

    # act
    context_emb = retriever.encode_context("fake_context")
    query_emb = retriever.encode_query("fake_query")

    # assert
    assert context_emb == [1, 2, 3]
    assert query_emb == [4, 5, 6]
