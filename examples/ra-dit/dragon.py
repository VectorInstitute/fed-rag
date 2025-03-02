"""Dragon Retriever"""

from typing import Optional

from pydantic import BaseModel, PrivateAttr
from sentence_transformers import SentenceTransformer
from torch import Tensor


class DragonRetriever(BaseModel):
    _encoder: Optional[SentenceTransformer] = PrivateAttr(default=None)
    _query_encoder: Optional[SentenceTransformer] = PrivateAttr(default=None)
    _context_encoder: Optional[SentenceTransformer] = PrivateAttr(default=None)

    def __init__(
        self,
        encoder: Optional[SentenceTransformer] = None,
        query_encoder: Optional[SentenceTransformer] = None,
        context_encoder: Optional[SentenceTransformer] = None,
    ) -> None:
        """Init method."""

        if encoder is not None:
            if query_encoder is not None:
                raise ValueError(
                    "If `encoder` is supplied, then cannot provide `query_encoder` as well."
                )
            if context_encoder is not None:
                raise ValueError(
                    "If `encoder` is supplied, then cannot provide `query_encoder` as well."
                )
        else:
            if query_encoder is None or context_encoder is None:
                raise ValueError(
                    "If `encoder` is None, then must supply both a `query_encoder` and a `context_encoder`."
                )

        super().__init__()
        self._encoder = encoder
        self._query_encoder = query_encoder
        self._context_encoder = context_encoder

    def encode_query(self, query: str | list[str]) -> Tensor:
        encoder: SentenceTransformer = (
            self._query_encoder if self._query_encoder else self._encoder
        )
        return encoder.encode(query)

    def encode_context(self, context: str | list[str]) -> Tensor:
        encoder: SentenceTransformer = (
            self._context_encoder if self._context_encoder else self._encoder
        )
        return encoder.encode(context)


if __name__ == "__main__":
    dragon_retriever = DragonRetriever(
        query_encoder=SentenceTransformer("nthakur/dragon-plus-query-encoder"),
        context_encoder=SentenceTransformer(
            "nthakur/dragon-plus-context-encoder"
        ),
    )

    query = "Where was Marie Curie born?"
    contexts = [
        "Maria Sklodowska, later known as Marie Curie, was born on November 7, 1867.",
        "Born in Paris on 15 May 1859, Pierre Curie was the son of Eugène Curie, a doctor of French Catholic origin from Alsace.",
    ]

    query_embeddings = dragon_retriever.encode_query(query)
    context_embeddings = dragon_retriever.encode_context(contexts)

    scores = query_embeddings @ context_embeddings.T
    print(scores)
