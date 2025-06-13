from .core import FedRAGError


class RAGTrainerManagerError(FedRAGError):
    """Base errors for all rag trainer manager relevant exceptions."""

    pass


class UnspecifiedRetrieverTrainer(RAGTrainerManagerError):
    """Raised if a retriever trainer has not been specified when one was expected to be."""

    pass


class UnspecifiedGeneratorTrainer(RAGTrainerManagerError):
    """Raised if a generator trainer has not been specified when one was expected to be."""

    pass


class UnsupportedTrainerMode(RAGTrainerManagerError):
    """Raised if an unsupported trainer mode has been supplied."""

    pass


class InconsistentRAGSystems(RAGTrainerManagerError):
    """Raised if trainers have inconsistent underlying RAG systems."""

    pass
