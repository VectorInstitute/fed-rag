from .core import FedRAGError


class TrainerError(FedRAGError):
    """Base errors for all rag trainer relevant exceptions."""

    pass


class InconsistentDatasetError(TrainerError):
    """Raised if underlying datasets between dataloaders are inconsistent."""

    pass


class InvalidLossError(TrainerError):
    """Raised if an unexpected loss is attached to a trainer object."""

    pass


class InvalidDataCollatorError(TrainerError):
    """Raised if an invalid data collator is attached to a trainer object."""

    pass


class MissingInputTensor(TrainerError):
    """Raised if a required tensor has not been supplied in the inputs."""

    pass
