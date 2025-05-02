from .core import FedRAGError


class TrainerError(FedRAGError):
    """Base errors for all rag trainer relevant exceptions."""

    pass


class InconsistentDatasetError(TrainerError):
    pass


class InvalidLossError(TrainerError):
    pass


class InvalidDataCollatorError(TrainerError):
    pass


class MissingInputTensor(TrainerError):
    pass
