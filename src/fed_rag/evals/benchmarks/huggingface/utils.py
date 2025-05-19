from importlib.util import find_spec

from fed_rag.exceptions import MissingExtraError


def check_huggingface_evals_installed(cls_name: str | None = None) -> None:
    datasets_spec = find_spec("datasets")

    has_huggingface = datasets_spec is not None

    if not has_huggingface:
        if cls_name:
            msg = (
                f"`{cls_name}` requires the `huggingface-evals` extra to be installed. "
                "To fix please run `pip install fed-rag[huggingface-evals]`."
            )
        else:
            msg = (
                "Missing installation of the huggingface-evals extra, yet is required "
                "by an import `HuggingFaceBenchmark` class. To fix please run "
                "`pip install fed-rag[huggingface-evals]`."
            )

        raise MissingExtraError(msg)
