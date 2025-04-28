from .core import FedRAGError


class RAGTrainerError(FedRAGError):
    """Base errors for all rag trainer relevant exceptions."""

    pass


class UnspecifiedRetrieverTrainer(RAGTrainerError):
    pass


class UnspecifiedGeneratorTrainer(RAGTrainerError):
    pass


class UnsupportedTrainerMode(RAGTrainerError):
    pass
