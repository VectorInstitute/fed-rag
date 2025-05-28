"""Tests for HellaSwag benchmark"""

from unittest.mock import MagicMock, patch

import pytest
from datasets import Dataset

import fed_rag.evals.benchmarks as benchmarks
from fed_rag.data_structures.evals import BenchmarkExample


@pytest.fixture
def dummy_hellaswag() -> Dataset:
    """Create a dummy HellaSwag dataset for testing."""
    return Dataset.from_dict(
        {
            "ind": [4],
            "activity_label": ["Removing ice from car"],
            "ctx_a": [
                "Then, the man writes over the snow covering the window of a car, and a woman wearing winter clothes smiles."
            ],
            "ctx_b": ["then"],
            "ctx": [
                "Then, the man writes over the snow covering the window of a car, and a woman wearing winter clothes smiles. then"
            ],
            "endings": [
                [
                    ", the man adds wax to the windshield and cuts it.",
                    ", a person board a ski lift, while two men supporting the head of the person wearing winter clothes snow as the we girls sled.",
                    ", the man puts on a christmas coat, knitted with netting.",
                    ", the man continues removing the snow on his car.",
                ]
            ],
            "source_id": ["activitynet~v_-1IBHYS3L-Y"],
            "split": ["train"],
            "split_type": ["indomain"],
            "label": ["3"],
        }
    )


@patch("datasets.load_dataset")
def test_hellaswag_benchmark(
    mock_load_dataset: MagicMock, dummy_hellaswag: Dataset
) -> None:
    mock_load_dataset.return_value = dummy_hellaswag
    hellaswag = benchmarks.HuggingFaceHellaSwag()

    assert isinstance(hellaswag[0], BenchmarkExample)
    assert (
        hellaswag[0].query
        == "Then, the man writes over the snow covering the window of a car, and a woman wearing winter clothes smiles. then"
    )
    assert hellaswag[0].response == "3"
    expected_context = (
        "0: , the man adds wax to the windshield and cuts it.\n"
        "1: , a person board a ski lift, while two men supporting the head of the person wearing winter clothes snow as the we girls sled.\n"
        "2: , the man puts on a christmas coat, knitted with netting.\n"
        "3: , the man continues removing the snow on his car."
    )
    assert hellaswag[0].context == expected_context


@patch("datasets.load_dataset")
def test_hellaswag_endings_structure(
    mock_load_dataset: MagicMock, dummy_hellaswag: Dataset
) -> None:
    mock_load_dataset.return_value = dummy_hellaswag
    hellaswag = benchmarks.HuggingFaceHellaSwag()
    endings_field = hellaswag[0].context
    endings_lines = endings_field.strip().split("\n")
    assert len(endings_lines) == 4
    assert all(isinstance(line, str) and line for line in endings_lines)


@patch("datasets.load_dataset")
def test_hellaswag_label_within_range(
    mock_load_dataset: MagicMock, dummy_hellaswag: Dataset
) -> None:
    mock_load_dataset.return_value = dummy_hellaswag
    hellaswag = benchmarks.HuggingFaceHellaSwag()
    # Response should be between 0 and 3 for standard HellaSwag
    response = int(hellaswag[0].response)
    assert 0 <= response <= 3
