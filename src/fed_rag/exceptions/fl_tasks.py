"""Exceptions for FL Tasks."""

from .core import FedRAGError


class FLTaskError(FedRAGError):
    """Base fl task error for all fl-task-related exceptions."""

    pass


class MissingFLTaskConfig(FLTaskError):
    """Raised if fl task `trainer` and `tester` do not have `__fl_task_tester_config` attr set."""

    pass


class MissingRequiredNetParam(FLTaskError):
    """Raised when invoking fl_task.server without passing the specified model/net param."""

    pass


class NetTypeMismatch(FLTaskError):
    """Raised when a `trainer` and `tester` spec have differing `net_parameter_class_name`.

    This indicates that the these methods have different types for the `net_parameter`.
    """

    pass
