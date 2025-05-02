import importlib.util

from fed_rag.exceptions.common import MissingExtraError

from .lsr import DataCollatorForLSR
from .ralt import DataCollatorForRALT

# check if huggingface extra is installed
_has_huggingface = (importlib.util.find_spec("transformers") is not None) and (
    importlib.util.find_spec("peft") is not None
)
if not _has_huggingface:
    msg = (
        f"`{__name__}` requires `huggingface` extra to be installed."
        " To fix please run `pip install fed-rag[huggingface]`."
    )
    raise MissingExtraError(msg)

__all__ = ["DataCollatorForLSR", "DataCollatorForRALT"]
