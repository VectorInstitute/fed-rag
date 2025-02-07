class MissingFLTaskConfig(Exception):
    pass


class MissingRequiredNetParam(Exception):
    """Raised when invoking fl_task.server without passing the specified model/net param."""

    pass
