from fed_rag.base.retriever import BaseRetriever
from fed_rag.retrievers.hf_sentence_transformer import (
    HFSentenceTransformerRetriever,
)


def test_hf_pretrained_generator_class() -> None:
    names_of_base_classes = [
        b.__name__ for b in HFSentenceTransformerRetriever.__mro__
    ]
    assert BaseRetriever.__name__ in names_of_base_classes
