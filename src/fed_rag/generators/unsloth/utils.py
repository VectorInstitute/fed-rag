from importlib.util import find_spec

from fed_rag.exceptions import MissingExtraError


def check_unsloth_installed(cls_name: str | None = None) -> None:
    unsloth_spec = find_spec("unsloth")

    has_unsloth = unsloth_spec is not None
    if not has_unsloth:
        if cls_name:
            msg = (
                f"`{cls_name}` requires the `unsloth` extra to be installed. "
                "To fix please run `pip install fed-rag[unsloth]`."
            )
        else:
            msg = (
                "Missing installation of the `unsloth` extra, yet is required "
                "by an imported class. To fix please run `pip install fed-rag[unsloth]`."
            )

        raise MissingExtraError(msg)
