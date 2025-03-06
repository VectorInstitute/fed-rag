from contextlib import nullcontext as does_not_raise

import torch

from fed_rag.base.retriever import BaseRetriever


def test_base_abstract_attr() -> None:
    abstract_methods = BaseRetriever.__abstractmethods__

    assert "encode_context" in abstract_methods
    assert "encode_query" in abstract_methods
    assert "encoder" in abstract_methods
    assert "query_encoder" in abstract_methods
    assert "context_encoder" in abstract_methods


def test_base_encode(mock_retriever: BaseRetriever) -> None:
    encoded_ctx = mock_retriever.encode_context("mock context")
    encoded_query = mock_retriever.encode_query("mock query")
    cosine_sim = encoded_ctx @ encoded_query.T
    *_, final_layer = mock_retriever.encoder.parameters()

    with does_not_raise():
        # cosine sim should be a Tensor with a single item
        cosine_sim.item()

    assert encoded_ctx.numel() == final_layer.size()[-1]
    assert encoded_query.numel() == final_layer.size()[-1]
    assert isinstance(mock_retriever.encoder, torch.nn.Module)
    assert mock_retriever.query_encoder is None
    assert mock_retriever.context_encoder is None
