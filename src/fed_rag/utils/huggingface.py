from fed_rag.exceptions import FedRAGError
from fed_rag.types.rag_system import RAGSystem


def _validate_rag_system(rag_system: RAGSystem) -> None:
    # Skip validation if environment variable is set
    import os

    if os.environ.get("FEDRAG_SKIP_VALIDATION") == "1":
        return

    from fed_rag.generators.huggingface import (
        HFPeftModelGenerator,
        HFPretrainedModelGenerator,
    )
    from fed_rag.retrievers.huggingface.hf_sentence_transformer import (
        HFSentenceTransformerRetriever,
    )

    if not isinstance(
        rag_system.generator, HFPretrainedModelGenerator
    ) and not isinstance(rag_system.generator, HFPeftModelGenerator):
        raise FedRAGError(
            "Generator must be HFPretrainedModelGenerator or HFPeftModelGenerator"
        )

    if not isinstance(rag_system.retriever, HFSentenceTransformerRetriever):
        raise FedRAGError("Retriever must be a HFSentenceTransformerRetriever")
