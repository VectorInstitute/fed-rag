from importlib.util import find_spec

from fed_rag.exceptions import MissingExtraError


def check_huggingface_installed(cls_name: str | None = None) -> None:
    transformers_spec = find_spec("transformers")
    peft_spec = find_spec("peft")
    sentence_transformers_spec = find_spec("sentence_transformers")

    has_huggingface = (
        (transformers_spec is not None)
        and (peft_spec is not None)
        and (sentence_transformers_spec is not None)
    )
    if not has_huggingface:
        if cls_name:
            msg = (
                f"`{cls_name}` requires the `huggingface` extra to be installed. "
                "To fix please run `pip install fed-rag[huggingface]`."
            )
        else:
            msg = (
                "Missing installation of the huggingface extra, yet is required "
                "by an imported class. To fix please run `pip install fed-rag[huggingface]`."
            )

        raise MissingExtraError(msg)
