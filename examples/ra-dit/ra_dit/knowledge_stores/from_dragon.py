"""Knowledge Store."""

# ra_dit
from ra_dit.retrievers.dragon import retriever

from .utils import knowledge_store_from_retriever

knowledge_store = knowledge_store_from_retriever(
    retriever=retriever, persist=True, name="dragon", overwrite=True
)
