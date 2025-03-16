class MissingFLTaskConfig(Exception):
    pass


class MissingRequiredNetParam(Exception):
    """Raised when invoking fl_task.server without passing the specified model/net param."""

    pass


class NetTypeMismatch(Exception):
    """Raised when a `trainer` and `tester` spec have differing `net_parameter_class_name`.

    This indicates that the these methods have different types for the `net_parameter`.
    """

    pass
