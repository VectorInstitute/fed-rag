"""RA-DIT original RAG System."""

from transformers.generation.utils import GenerationConfig
from transformers.utils.quantization_config import BitsAndBytesConfig

from fed_rag.generators.hf_pretrained_model import HFPretrainedModelGenerator
from fed_rag.retrievers.hf_sentence_transformer import (
    HFSentenceTransformerRetriever,
)
from fed_rag.types.rag_system import RAGConfig, RAGSystem

from .knowledge_store import knowledge_store


def main(model_name: str) -> RAGSystem:
    """Build RAG System."""

    ## retriever
    dragon_retriever = HFSentenceTransformerRetriever(
        query_model_name="nthakur/dragon-plus-query-encoder",
        context_model_name="nthakur/dragon-plus-context-encoder",
        load_model_kwargs={"device": "cpu"},
        load_model_at_init=False,
    )

    ## generator
    generation_cfg = GenerationConfig(
        do_sample=True,
        eos_token_id=[128000, 128009],
        bos_token_id=128000,
        max_new_tokens=4096,
        top_p=0.9,
        temperature=0.6,
        cache_implementation="offloaded",
        stop_strings="</response>",
    )
    # quantization_config = BitsAndBytesConfig(load_in_8bit=True)
    llama3_generator = HFPretrainedModelGenerator(
        model_name=model_name,
        load_model_kwargs={
            # "quantization_config": quantization_config,
            "device_map": "cpu",
        },
        generation_config=generation_cfg,
        load_model_at_init=False,
    )

    ## assemble
    rag_config = RAGConfig(top_k=2)
    rag_system = RAGSystem(
        knowledge_store=knowledge_store,  # knowledge store loaded from knowledge_store.py
        generator=llama3_generator,
        retriever=dragon_retriever,
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
