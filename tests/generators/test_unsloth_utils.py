import re
from unittest.mock import patch

import pytest

from fed_rag.exceptions import MissingExtraError
from fed_rag.generators.unsloth.utils import check_unsloth_installed


def test_check_raises_error() -> None:
    """Check raises error from utils."""

    modules = {"unsloth": None}

    with patch.dict("sys.modules", modules):
        # without class name
        msg = (
            "Missing installation of the `unsloth` extra, yet is required "
            "by an imported class. To fix please run `pip install fed-rag[unsloth]`."
        )
        with pytest.raises(
            MissingExtraError,
            match=re.escape(msg),
        ):
            check_unsloth_installed()

        # with class name
        msg = (
            "`FakeClass` requires the `unsloth` extra to be installed. "
            "To fix please run `pip install fed-rag[unsloth]`."
        )
        with pytest.raises(
            MissingExtraError,
            match=re.escape(msg),
        ):
            check_unsloth_installed("FakeClass")
