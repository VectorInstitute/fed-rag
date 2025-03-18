"""RA-DIT original RAG System."""

from typing import Literal

from ra_dit.generators import GENERATORS
from ra_dit.knowledge_stores import KNOWLEDGE_STORES
from ra_dit.retrievers import RETRIEVERS

from fed_rag.types.rag_system import RAGConfig, RAGSystem


def main(
    retriever_id: str,
    generator_id: str,
    generator_variant: Literal["plain", "lora", "qlora"],
) -> RAGSystem:
    """Build RAG System."""

    retriever = RETRIEVERS[retriever_id]
    knowledge_store = KNOWLEDGE_STORES[f"from_{retriever_id}"]
    generator = GENERATORS[generator_id][generator_variant]

    ## assemble
    rag_config = RAGConfig(top_k=2)
    rag_system = RAGSystem(
        knowledge_store=knowledge_store,  # knowledge store loaded from knowledge_store.py
        generator=generator,
        retriever=retriever,
        rag_config=rag_config,
    )

    return rag_system


if __name__ == "__main__":
    import fire

    rag_system: RAGSystem = fire.Fire(main)

    ## use the rag_system
    source_nodes = rag_system.retrieve("What is a Tulip?")
    response = rag_system.query("What is a Tulip?")

    print(source_nodes[0].score)
    print(f"\n{response}")
