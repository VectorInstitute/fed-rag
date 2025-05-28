"""Tests for HotpotQA benchmark"""

import re
import sys
from unittest.mock import MagicMock, patch

import pytest
from datasets import Dataset

import fed_rag.evals.benchmarks as benchmarks
from fed_rag.data_structures.evals import BenchmarkExample
from fed_rag.exceptions import MissingExtraError


@pytest.fixture
def dummy_hotpotqa() -> Dataset:
    """Create a dummy HotpotQA dataset for testing."""
    return Dataset.from_dict(
        {
            "id": ["5a7a06935542990198eaf050"],
            "question": [
                "Which magazine was started first Arthur's Magazine or First for Women?"
            ],
            "answer": ["Arthur's Magazine"],
            "type": ["comparison"],
            "level": ["medium"],
            "supporting_facts": [
                {
                    "title": ["Arthur's Magazine", "First for Women"],
                    "sent_id": [0, 0],
                }
            ],
            "context": [
                {
                    "title": [
                        "Arthur's Magazine",
                        "First for Women",
                        "Other Magazine",
                    ],
                    "sentences": [
                        [
                            "Arthur's Magazine (1844–1846) was an American literary periodical.",
                            "It was founded by Timothy Shay Arthur.",
                        ],
                        [
                            "First for Women is a woman's magazine published by Bauer Media Group.",
                            "The magazine was started in 1989.",
                        ],
                        ["This is another magazine.", "It has some content."],
                    ],
                }
            ],
        }
    )


@patch("datasets.load_dataset")
def test_hotpotqa_benchmark(mock_load_dataset: MagicMock) -> None:
    # arrange & act
    hotpotqa = benchmarks.HuggingFaceHotpotQA()

    # assert
    mock_load_dataset.assert_called_once_with(
        benchmarks.HuggingFaceHotpotQA.dataset_name,
        name=hotpotqa.configuration_name,
        split=hotpotqa.split,
        streaming=hotpotqa.streaming,
    )


@patch("datasets.load_dataset")
def test_hotpotqa_query_response_context_extractors(
    mock_load_dataset: MagicMock, dummy_hotpotqa: Dataset
) -> None:
    # arrange
    mock_load_dataset.return_value = dummy_hotpotqa
    hotpotqa = benchmarks.HuggingFaceHotpotQA()

    # assert
    assert isinstance(hotpotqa[0], BenchmarkExample)
    assert (
        hotpotqa[0].query
        == "Which magazine was started first Arthur's Magazine or First for Women?"
    )
    assert hotpotqa[0].response == "Arthur's Magazine"

    expected_context = (
        "Arthur's Magazine: Arthur's Magazine (1844–1846) was an American literary periodical. "
        "It was founded by Timothy Shay Arthur. "
        "First for Women: First for Women is a woman's magazine published by Bauer Media Group. "
        "The magazine was started in 1989. "
        "Other Magazine: This is another magazine. It has some content."
    )
    assert hotpotqa[0].context == expected_context


@patch("datasets.load_dataset")
def test_hotpotqa_no_context(mock_load_dataset: MagicMock) -> None:
    """Test handling when context is missing or not a dict."""
    dataset = Dataset.from_dict(
        {
            "id": ["1"],
            "question": ["Test question?"],
            "answer": ["Test answer"],
            "type": ["comparison"],
            "level": ["easy"],
            "supporting_facts": [{"title": [], "sent_id": []}],
            "context": ["This is not a dict"],  # Invalid context format
        }
    )
    mock_load_dataset.return_value = dataset
    hotpotqa = benchmarks.HuggingFaceHotpotQA()
    assert hotpotqa[0].context is None


@patch("datasets.load_dataset")
def test_hotpotqa_empty_context(mock_load_dataset: MagicMock) -> None:
    """Test handling when context has empty title/sentences."""
    dataset = Dataset.from_dict(
        {
            "id": ["1"],
            "question": ["Test question?"],
            "answer": ["Test answer"],
            "type": ["comparison"],
            "level": ["easy"],
            "supporting_facts": [{"title": [], "sent_id": []}],
            "context": [{"title": [], "sentences": []}],
        }
    )
    mock_load_dataset.return_value = dataset
    hotpotqa = benchmarks.HuggingFaceHotpotQA()
    assert hotpotqa[0].context is None


@patch("datasets.load_dataset")
def test_hotpotqa_missing_context_key(mock_load_dataset: MagicMock) -> None:
    """Test handling when example has no context key."""
    dataset = Dataset.from_dict(
        {
            "id": ["1"],
            "question": ["Test question?"],
            "answer": ["Test answer"],
            "type": ["comparison"],
            "level": ["easy"],
            "supporting_facts": [{"title": [], "sent_id": []}]
            # No context key
        }
    )
    mock_load_dataset.return_value = dataset
    hotpotqa = benchmarks.HuggingFaceHotpotQA()
    assert hotpotqa[0].context is None


def test_huggingface_evals_extra_missing() -> None:
    """Test that proper error is raised when huggingface-evals extra is missing."""
    modules = {
        "datasets": None,
    }
    module_to_import = "fed_rag.evals.benchmarks"
    original_module = sys.modules.pop(module_to_import, None)

    with patch.dict("sys.modules", modules):
        msg = (
            "`HuggingFaceHotpotQA` requires the `huggingface-evals` extra to be installed. "
            "To fix please run `pip install fed-rag[huggingface-evals]`."
        )
        with pytest.raises(
            MissingExtraError,
            match=re.escape(msg),
        ):
            import fed_rag.evals.benchmarks as benchmarks

            benchmarks.HuggingFaceHotpotQA()

    # restore module so to not affect other tests
    if original_module:
        sys.modules[module_to_import] = original_module
