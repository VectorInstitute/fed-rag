import re
import sys
from unittest.mock import MagicMock, patch

import pytest
from datasets import Dataset

import fed_rag.evals.benchmarks as benchmarks
from fed_rag.data_structures.evals import BenchmarkExample
from fed_rag.exceptions import MissingExtraError


@patch("datasets.load_dataset")
def test_mmlu_benchmark(mock_load_dataset: MagicMock) -> None:
    # arrange
    mmlu = benchmarks.HuggingFaceMMLU()

    mock_load_dataset.assert_called_once_with(
        benchmarks.HuggingFaceMMLU.dataset_name,
        name=mmlu.configuration_name,
        split=mmlu.split,
        streaming=mmlu.streaming,
    )


@patch("datasets.load_dataset")
def test_mmlu_query_response_context_extractors(
    mock_load_dataset: MagicMock, dummy_mmlu: Dataset
) -> None:
    # arrange
    mock_load_dataset.return_value = dummy_mmlu
    mmlu = benchmarks.HuggingFaceMMLU()

    assert mmlu.num_examples == 1
    assert isinstance(mmlu[0], BenchmarkExample)
    assert (
        mmlu[0].query
        == "What is the embryological origin of the hyoid bone?\n\nA: The first pharyngeal arch\nB: The first and second pharyngeal arches\nC: The second pharyngeal arch\nD: The second and third pharyngeal arches"
    )


def test_huggingface_evals_extra_missing() -> None:
    modules = {
        "datasets": None,
    }
    module_to_import = "fed_rag.evals.benchmarks"
    original_module = sys.modules.pop(module_to_import, None)

    with patch.dict("sys.modules", modules):
        msg = (
            "`HuggingFaceMMLU` requires the `huggingface-evals` extra to be installed. "
            "To fix please run `pip install fed-rag[huggingface-evals]`."
        )
        with pytest.raises(
            MissingExtraError,
            match=re.escape(msg),
        ):
            import fed_rag.evals.benchmarks as benchmarks

            benchmarks.HuggingFaceMMLU()

    # restore module so to not affect other tests
    if original_module:
        sys.modules[module_to_import] = original_module
