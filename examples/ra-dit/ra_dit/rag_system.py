"""RA-DIT original RAG System."""

import logging
from typing import Literal

from ra_dit.generators import GENERATORS
from ra_dit.knowledge_stores.utils import knowledge_store_from_retriever
from ra_dit.retrievers import RETRIEVERS

from fed_rag.types.rag_system import RAGConfig, RAGSystem

logger = logging.getLogger("ra_dit.rag_system")


def main(
    retriever_id: str,
    generator_id: str,
    generator_variant: Literal["plain", "lora", "qlora"],
    retriever_checkpoint_path: str | None = None,
    generator_checkpoint_path: str | None = None,
) -> RAGSystem:
    """Build RAG System."""

    retriever = RETRIEVERS[retriever_id]
    logger.info(f"Loaded retriever: {retriever_id}")
    if retriever_checkpoint_path:
        # update model name to checkpoint path
        retriever.model_name = retriever_checkpoint_path
        logger.info(
            f"Updated retriever to upload checkpoint in: {retriever_checkpoint_path}"
        )

    knowledge_store = knowledge_store_from_retriever(
        retriever=retriever,
        name=retriever_id,
    )
    logger.info(
        f"Successfully loaded knowledge from retriever: {retriever_id}"
    )

    generator = GENERATORS[generator_id][generator_variant]
    logger.info(f"Loaded generator: {generator_id}")
    if generator_checkpoint_path:
        # update model name to checkpoint path
        generator.model_name = generator_checkpoint_path
        logger.info(
            f"Updated generator to upload checkpoint in: {generator_checkpoint_path}"
        )

    ## assemble
    rag_config = RAGConfig(top_k=1)
    rag_system = RAGSystem(
        knowledge_store=knowledge_store,  # knowledge store loaded from knowledge_store.py
        generator=generator,
        retriever=retriever,
        rag_config=rag_config,
    )

    return rag_system


if __name__ == "__main__":
    import fire

    from fed_rag.generators.huggingface import HFPeftModelGenerator

    rag_system: RAGSystem = fire.Fire(main)

    if isinstance(rag_system.generator, HFPeftModelGenerator):
        rag_system.generator.model.merge_and_unload()

    ## use the rag_system
    source_nodes = rag_system.retrieve("What is a Tulip?")
    response = rag_system.query("What is a Tulip?")

    print(source_nodes[0].score)
    print(f"\n{response}")
