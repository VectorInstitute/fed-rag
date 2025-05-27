"""Common exceptions."""

from .core import FedRAGError


class MissingExtraError(FedRAGError):
    """Raised when a fed-rag extra is not installed."""


class IncompatibleVersionError(FedRAGError):
    """Raised when a fed-rag component is not compatible with the current version."""
