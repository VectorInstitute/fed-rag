"""HuggingFace SentenceTransformer Retriever"""

from typing import Any, Literal, TypedDict, cast

import torch
from pydantic import ConfigDict, Field, PrivateAttr
from sentence_transformers import SentenceTransformer
from typing_extensions import Self

from fed_rag.base.retriever import BaseRetriever


class LoadKwargs(TypedDict):
    model: dict
    query: dict
    context: dict

    @classmethod  # type: ignore[misc]
    def default(cls) -> Self:
        return cls(model={}, query={}, context={})


class InvalidLoadType(Exception):
    pass


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
    load_model_kwargs: LoadKwargs = Field(
        description="Optional kwargs dict for loading models from HF. Defaults to None.",
        default_factory=LoadKwargs.default,  # type: ignore[attr-defined]
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
            if model_name:
                self._encoder = self._load_model_from_hf(load_type="encoder")
            else:
                self._query_encoder = self._load_model_from_hf(
                    load_type="query_encoder"
                )
                self._context_encoder = self._load_model_from_hf(
                    load_type="context_encoder"
                )

    def _load_model_from_hf(
        self,
        load_type: Literal["encoder", "query_encoder", "context_encoder"],
        **kwargs: Any,
    ) -> SentenceTransformer:
        if load_type == "encoder":
            load_kwargs = self.load_model_kwargs["model"]
            load_kwargs.update(kwargs)
            return SentenceTransformer(self.model_name, **kwargs)
        elif load_type == "context_encoder":
            load_kwargs = self.load_model_kwargs["context"]
            load_kwargs.update(kwargs)
            return SentenceTransformer(self.context_model_name, **kwargs)
        elif load_type == "query_encoder":
            load_kwargs = self.load_model_kwargs["query"]
            load_kwargs.update(kwargs)
            return SentenceTransformer(self.query_model_name, **kwargs)
        else:
            raise InvalidLoadType("Invalid `load_type` supplied.")

    def encode_context(
        self, context: str | list[str], **kwargs: Any
    ) -> torch.Tensor:
        # validation guarantees one of these is not None
        encoder = self.encoder if self.encoder else self.context_encoder
        encoder = cast(SentenceTransformer, encoder)

        return encoder.encode(context)

    def encode_query(
        self, query: str | list[str], **kwargs: Any
    ) -> torch.Tensor:
        # validation guarantees one of these is not None
        encoder = self.encoder if self.encoder else self.query_encoder
        encoder = cast(SentenceTransformer, encoder)

        return encoder.encode(query)

    @property
    def encoder(self) -> SentenceTransformer | None:
        if self.model_name and self._encoder is None:
            self._encoder = self._load_model_from_hf(load_type="encoder")
        return self._encoder

    @property
    def query_encoder(self) -> SentenceTransformer | None:
        if self.query_model_name and self._query_encoder is None:
            self._query_encoder = self._load_model_from_hf(
                load_type="query_encoder"
            )
        return self._query_encoder

    @property
    def context_encoder(self) -> SentenceTransformer | None:
        if self.context_model_name and self._context_encoder is None:
            self._context_encoder = self._load_model_from_hf(
                load_type="context_encoder"
            )
        return self._context_encoder
