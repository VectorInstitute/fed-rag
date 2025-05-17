import re
from unittest.mock import patch

import pytest

from fed_rag.exceptions import MissingExtraError
from fed_rag.generators.huggingface.utils import check_huggingface_installed


def test_check_raises_error() -> None:
    """Check raises error from utils."""

    modules = {"transformers": None}

    with patch.dict("sys.modules", modules):
        msg = (
            "Missing installation of the huggingface extra, yet is required "
            "by an imported class. To fix please run `pip install fed-rag[huggingface]`."
        )
        with pytest.raises(
            MissingExtraError,
            match=re.escape(msg),
        ):
            check_huggingface_installed()
