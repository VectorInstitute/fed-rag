"""Exceptions for Evals."""

from .core import FedRAGError, FedRAGWarning


class EvalsError(FedRAGError):
    """Base evals error for all evals-related exceptions."""

    pass


class EvalsWarning(FedRAGWarning):
    """Base inspector warning for all evals-related warnings."""

    pass


class BenchmarkGetExamplesError(EvalsError):
    """Raised if an error occurs when getting examples for a benchmark."""

    pass


class BenchmarkParseError(EvalsError):
    """Raised when errors occur during parsing examples."""

    pass


class EvaluationsFileNotFoundError(EvalsError, FileNotFoundError):
    """Benchmark evaluations file not found error."""

    pass
