from inspect import Parameter
from typing import cast


def get_type_name(t: Parameter) -> str | None:
    if isinstance(t.annotation, str):
        type_name = t.annotation
    else:
        type_name = getattr(
            t.annotation, "__name__", None
        )  # type:ignore [assignment]
    type_name = cast(str | None, type_name)  # type:ignore [assignment]
    return type_name
