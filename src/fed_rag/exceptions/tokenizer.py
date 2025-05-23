"""Exceptions for Tokenizer."""

from .core import FedRAGError, FedRAGWarning


class TokenizerError(FedRAGError):
    """Base evals error for all tokenizer-related exceptions."""

    pass


class TokenizerWarning(FedRAGWarning):
    """Base inspector warning for all tokenizer-related warnings."""

    pass
