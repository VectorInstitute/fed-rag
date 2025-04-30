from .core import FedRAGError


class TrainerError(FedRAGError):
    """Base errors for all rag trainer relevant exceptions."""

    pass


class InvalidLossError(TrainerError):
    pass


class MissingInputTensor(TrainerError):
    pass
