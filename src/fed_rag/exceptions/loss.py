"""Exceptions for loss."""


class LossError(Exception):
    """Base loss errors for all loss-related exceptions."""

    pass


class InvalidReductionParam(LossError):
    pass
