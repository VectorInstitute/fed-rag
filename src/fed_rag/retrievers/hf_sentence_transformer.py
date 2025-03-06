"""HuggingFace SentenceTransformer Retriever"""

from typing import Any

from pydantic import ConfigDict, Field, PrivateAttr
from sentence_transformers import SentenceTransformer

from fed_rag.base.retriever import BaseRetriever


class HFSentenceTransformerRetriever(BaseRetriever):
    model_config = ConfigDict(protected_namespaces=("pydantic_model_",))
    model_name: str | None = Field(
        description="Name of HuggingFace SentenceTransformer model.",
        default=None,
    )
    query_model_name: str | None = Field(
        description="Name of HuggingFace SentenceTransformer model used for encoding queries.",
        default=None,
    )
    context_model_name: str | None = Field(
        description="Name of HuggingFace SentenceTransformer model used for encoding context.",
        default=None,
    )
    _encoder: SentenceTransformer | None = PrivateAttr(default=None)
    _query_encoder: SentenceTransformer | None = PrivateAttr(default=None)
    _context_encoder: SentenceTransformer | None = PrivateAttr(default=None)

    def __init__(
        self,
        model_name: str | None = None,
        query_model_name: str | None = None,
        context_model_name: str | None = None,
        load_model_at_init: bool = True,
    ):
        super().__init__(
            model_name=model_name,
            query_model_name=query_model_name,
            context_model_name=context_model_name,
        )
        if load_model_at_init:
            ...

    def _load_model_from_hf(self, **kwargs: Any) -> SentenceTransformer:
        return SentenceTransformer(self.model_name)
