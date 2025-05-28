"""Tests for SQuAD 2.0 benchmark"""

import re
import sys
from unittest.mock import MagicMock, patch

import pytest
from datasets import Dataset

import fed_rag.evals.benchmarks as benchmarks
from fed_rag.data_structures.evals import BenchmarkExample
from fed_rag.exceptions import MissingExtraError


@pytest.fixture
def dummy_squad_v2_answerable() -> Dataset:
    """Create a dummy SQuAD 2.0 dataset with answerable question."""
    return Dataset.from_dict(
        {
            "id": ["56ddde2d66d3e219004dad4d"],
            "title": ["Symbiosis"],
            "context": [
                "Symbiotic relationships include those associations in which one organism lives on another (ectosymbiosis, such as..."
            ],
            "question": ["What is an example of ectosymbiosis?"],
            "answers": [{"text": ["mistletoe"], "answer_start": [114]}],
        }
    )


@pytest.fixture
def dummy_squad_v2_unanswerable() -> Dataset:
    """Create a dummy SQuAD 2.0 dataset with unanswerable question."""
    return Dataset.from_dict(
        {
            "id": ["5ad3ed26604f3c001a3ff799"],
            "title": ["Normans"],
            "context": [
                "The Normans were the people who gave their name to Normandy, a region in France."
            ],
            "question": ["What is the population of Normandy?"],
            "answers": [{"text": [], "answer_start": []}],
        }
    )


@patch("datasets.load_dataset")
def test_squad_v2_benchmark(mock_load_dataset: MagicMock) -> None:
    # arrange & act
    squad = benchmarks.HuggingFaceSQuADv2()

    # assert
    mock_load_dataset.assert_called_once_with(
        benchmarks.HuggingFaceSQuADv2.dataset_name,
        name=squad.configuration_name,
        split=squad.split,
        streaming=squad.streaming,
    )


@patch("datasets.load_dataset")
def test_squad_v2_answerable_question(
    mock_load_dataset: MagicMock, dummy_squad_v2_answerable: Dataset
) -> None:
    # arrange
    mock_load_dataset.return_value = dummy_squad_v2_answerable
    squad = benchmarks.HuggingFaceSQuADv2()

    # assert
    assert isinstance(squad[0], BenchmarkExample)
    assert squad[0].query == "What is an example of ectosymbiosis?"
    assert squad[0].response == "mistletoe"
    assert (
        squad[0].context
        == "Symbiotic relationships include those associations in which one organism lives on another (ectosymbiosis, such as..."
    )


@patch("datasets.load_dataset")
def test_squad_v2_unanswerable_question(
    mock_load_dataset: MagicMock, dummy_squad_v2_unanswerable: Dataset
) -> None:
    # arrange
    mock_load_dataset.return_value = dummy_squad_v2_unanswerable
    squad = benchmarks.HuggingFaceSQuADv2()

    # assert
    assert squad[0].response == "[NO ANSWER]"


@patch("datasets.load_dataset")
def test_squad_v2_missing_answers_field(mock_load_dataset: MagicMock) -> None:
    """Test handling when answers field is missing."""
    dataset = Dataset.from_dict(
        {
            "id": ["1"],
            "title": ["Test"],
            "context": ["Test context"],
            "question": ["Test question?"]
            # No answers field
        }
    )
    mock_load_dataset.return_value = dataset
    squad = benchmarks.HuggingFaceSQuADv2()
    assert squad[0].response == "[NO ANSWER]"


def test_huggingface_evals_extra_missing() -> None:
    """Test that proper error is raised when huggingface-evals extra is missing."""
    modules = {
        "datasets": None,
    }
    module_to_import = "fed_rag.evals.benchmarks"
    original_module = sys.modules.pop(module_to_import, None)

    with patch.dict("sys.modules", modules):
        msg = (
            "`HuggingFaceSQuADv2` requires the `huggingface-evals` extra to be installed. "
            "To fix please run `pip install fed-rag[huggingface-evals]`."
        )
        with pytest.raises(
            MissingExtraError,
            match=re.escape(msg),
        ):
            import fed_rag.evals.benchmarks as benchmarks

            benchmarks.HuggingFaceSQuADv2()

    # restore module so to not affect other tests
    if original_module:
        sys.modules[module_to_import] = original_module
