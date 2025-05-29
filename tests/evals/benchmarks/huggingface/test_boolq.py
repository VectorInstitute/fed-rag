"""Tests for BoolQ benchmark"""

from unittest.mock import MagicMock, patch

import pytest
from datasets import Dataset

import fed_rag.evals.benchmarks as benchmarks
from fed_rag.data_structures.evals import BenchmarkExample


@pytest.fixture
def dummy_boolq() -> Dataset:
    """Create a dummy BoolQ dataset for testing."""
    return Dataset.from_dict(
        {
            "question": ["is confectionary sugar the same as powdered sugar"],
            "answer": [True],
            "passage": [
                "Powdered sugar, also called confectioners' sugar, is a finely ground sugar ..."
            ],
        }
    )


@patch("datasets.load_dataset")
def test_boolq_query_response_context_extractors(
    mock_load_dataset: MagicMock, dummy_boolq: Dataset
) -> None:
    mock_load_dataset.return_value = dummy_boolq
    boolq = benchmarks.HuggingFaceBoolQ()

    assert isinstance(boolq[0], BenchmarkExample)
    assert (
        boolq[0].query == "is confectionary sugar the same as powdered sugar"
    )
    assert boolq[0].response == "true"
    assert (
        boolq[0].context
        == "Powdered sugar, also called confectioners' sugar, is a finely ground sugar ..."
    )


@patch("datasets.load_dataset")
def test_boolq_false_response(mock_load_dataset: MagicMock) -> None:
    dataset = Dataset.from_dict(
        {
            "question": ["is elder scrolls online the same as skyrim"],
            "answer": [False],
            "passage": [
                "As with other games in The Elder Scrolls series, the game is set on the continent of Tamriel."
            ],
        }
    )
    mock_load_dataset.return_value = dataset
    boolq = benchmarks.HuggingFaceBoolQ()

    assert boolq[0].response == "false"
