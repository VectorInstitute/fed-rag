import re
from unittest.mock import patch

import pytest

from fed_rag.evals.benchmarks.huggingface.utils import (
    check_huggingface_evals_installed,
)
from fed_rag.exceptions import MissingExtraError


def test_check_raises_error() -> None:
    """Check raises error from utils."""

    modules = {"datasets": None}

    with patch.dict("sys.modules", modules):
        msg = (
            "Missing installation of the huggingface-evals extra, yet is required "
            "by an import `HuggingFaceBenchmark` class. To fix please run "
            "`pip install fed-rag[huggingface-evals]`."
        )

        with pytest.raises(
            MissingExtraError,
            match=re.escape(msg),
        ):
            check_huggingface_evals_installed()
