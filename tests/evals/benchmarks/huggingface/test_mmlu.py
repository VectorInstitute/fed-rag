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

    expected_query = mmlu.prompt_template.format(
        question="What is the embryological origin of the hyoid bone?",
        A="The first pharyngeal arch",
        B="The first and second pharyngeal arches",
        C="The second pharyngeal arch",
        D="The second and third pharyngeal arches",
    )
    assert mmlu[0].query == expected_query


@patch("datasets.load_dataset")
def test_mmlu_prompt_template_override(
    mock_load_dataset: MagicMock, dummy_mmlu: Dataset
) -> None:
    # arrange
    mock_load_dataset.return_value = dummy_mmlu

    custom_prompt = """{question}
A) {A}
B) {B}
C) {C}
D) {D}
Choose one option."""

    mmlu = benchmarks.HuggingFaceMMLU(prompt_template=custom_prompt)

    expected_query = custom_prompt.format(
        question="What is the embryological origin of the hyoid bone?",
        A="The first pharyngeal arch",
        B="The first and second pharyngeal arches",
        C="The second pharyngeal arch",
        D="The second and third pharyngeal arches",
    )
    assert mmlu[0].query == expected_query


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
