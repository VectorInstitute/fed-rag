from .core import FedRAGError


class RAGTrainerManagerError(FedRAGError):
    """Base errors for all rag trainer manager relevant exceptions."""

    pass


class UnspecifiedRetrieverTrainer(RAGTrainerManagerError):
    pass


class UnspecifiedGeneratorTrainer(RAGTrainerManagerError):
    pass


class UnsupportedTrainerMode(RAGTrainerManagerError):
    pass


class InconsistentRAGSystems(RAGTrainerManagerError):
    pass
