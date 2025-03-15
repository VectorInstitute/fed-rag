"""Knowledge Store."""

from pathlib import Path

# ra_dit
from ra_dit.retrievers.dragon import retriever

from .utils import knowledge_store_from_retriever

DATA_PATH = Path(__file__).parents[1].absolute() / "data"

knowledge_store = knowledge_store_from_retriever(retriever=retriever)


if __name__ == "__main__":
    print(knowledge_store.count)
