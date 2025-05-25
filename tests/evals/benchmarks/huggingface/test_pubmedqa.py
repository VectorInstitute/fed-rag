"""Tests for PubMedQA benchmark"""

import re
import sys
from unittest.mock import MagicMock, patch

import pytest
from datasets import Dataset

import fed_rag.evals.benchmarks as benchmarks
from fed_rag.data_structures.evals import BenchmarkExample
from fed_rag.exceptions import BenchmarkGetExamplesError, MissingExtraError


@pytest.fixture
def dummy_pubmedqa() -> Dataset:
    """Create a dummy PubMedQA dataset for testing."""
    return Dataset.from_dict(
        {
            "pubid": ["12345"],
            "question": [
                "Is increased time from neoadjuvant chemoradiation to surgery associated with higher pathologic complete response rates in esophageal cancer?"
            ],
            "context": [
                {
                    "contexts": ["Text A", "Text B", "Text C"],
                    "labels": ["OBJECTIVE", "METHODS", "RESULTS"],
                    "meshes": ["mesh1", "mesh2"],
                }
            ],
            "long_answer": [
                "Based on our analysis, increased time from neoadjuvant chemoradiation to surgery is associated with higher pathologic complete response rates."
            ],
            "final_decision": ["yes"],
        }
    )


@patch("datasets.load_dataset")
def test_pubmedqa_benchmark(mock_load_dataset: MagicMock) -> None:
    # arrange & act
    pubmedqa = benchmarks.HuggingFacePubMedQA()

    # assert
    mock_load_dataset.assert_called_once_with(
        benchmarks.HuggingFacePubMedQA.dataset_name,
        name=pubmedqa.configuration_name,
        split=pubmedqa.split,
        streaming=pubmedqa.streaming,
    )


@patch("datasets.load_dataset")
def test_pubmedqa_query_response_context_extractors(
    mock_load_dataset: MagicMock, dummy_pubmedqa: Dataset
) -> None:
    # arrange
    mock_load_dataset.return_value = dummy_pubmedqa
    pubmedqa = benchmarks.HuggingFacePubMedQA()

    assert isinstance(pubmedqa[0], BenchmarkExample)
    assert (
        pubmedqa[0].query
        == "Is increased time from neoadjuvant chemoradiation to surgery associated with higher pathologic complete response rates in esophageal cancer?"
    )

    assert pubmedqa[0].response == "yes"

    expected_context = " ".join(dummy_pubmedqa[0]["context"]["contexts"])

    assert pubmedqa[0].context == expected_context


@patch("datasets.load_dataset")
def test_pubmedqa_context_fallback(mock_load_dataset: MagicMock) -> None:
    dataset = Dataset.from_dict(
        {
            "pubid": ["1"],
            "question": ["Test question?"],
            "context": [
                {"foo": ["a", "b"], "bar": "baz"}
            ],  # No "contexts" key
            "long_answer": ["Test answer"],
            "final_decision": ["yes"],
        }
    )
    mock_load_dataset.return_value = dataset
    pubmedqa = benchmarks.HuggingFacePubMedQA()
    # Should join both lists and strings together
    assert set(pubmedqa[0].context.split()) == set("a b baz".split())


@patch("datasets.load_dataset")
def test_pubmedqa_context_is_string(mock_load_dataset: MagicMock) -> None:
    dataset = Dataset.from_dict(
        {
            "pubid": ["1"],
            "question": ["Test question?"],
            "context": ["This is just a string context."],
            "long_answer": ["Test answer"],
            "final_decision": ["yes"],
        }
    )
    mock_load_dataset.return_value = dataset
    pubmedqa = benchmarks.HuggingFacePubMedQA()
    assert pubmedqa[0].context == "This is just a string context."


@patch("datasets.load_dataset")
def test_pubmedqa_different_response_types(
    mock_load_dataset: MagicMock,
) -> None:
    """Test handling of different response types (yes/no/maybe)."""
    diverse_dataset = Dataset.from_dict(
        {
            "pubid": ["1", "2", "3"],
            "question": ["Q1?", "Q2?", "Q3?"],
            "context": [
                {
                    "contexts": ["Context 1"],
                    "labels": ["LABEL1"],
                    "meshes": ["meshA"],
                },
                {
                    "contexts": ["Context 2"],
                    "labels": ["LABEL2"],
                    "meshes": ["meshB"],
                },
                {
                    "contexts": ["Context 3"],
                    "labels": ["LABEL3"],
                    "meshes": ["meshC"],
                },
            ],
            "long_answer": ["Answer 1", "Answer 2", "Answer 3"],
            "final_decision": ["yes", "no", "maybe"],
        }
    )

    mock_load_dataset.return_value = diverse_dataset
    pubmedqa = benchmarks.HuggingFacePubMedQA()

    assert pubmedqa[0].response == "yes"
    assert pubmedqa[1].response == "no"
    assert pubmedqa[2].response == "maybe"


@patch("datasets.load_dataset")
def test_pubmedqa_context_unexpected_type(
    mock_load_dataset: MagicMock,
) -> None:
    dataset = Dataset.from_dict(
        {
            "pubid": ["1"],
            "question": ["Test question?"],
            "context": [1234],  # Not a dict, list, or str
            "long_answer": ["Test answer"],
            "final_decision": ["yes"],
        }
    )

    mock_load_dataset.return_value = dataset

    with pytest.raises(
        BenchmarkGetExamplesError,
        match=re.escape(
            "Failed to parse examples: Unexpected context type: <class 'int'> "
            "in example: {'pubid': '1', 'question': 'Test question?', 'context': 1234, "
            "'long_answer': 'Test answer', 'final_decision': 'yes'}"
        ),
    ):
        _ = benchmarks.HuggingFacePubMedQA()[0].context


def test_huggingface_evals_extra_missing() -> None:
    """Test that proper error is raised when huggingface-evals extra is missing."""
    modules = {
        "datasets": None,
    }
    module_to_import = "fed_rag.evals.benchmarks"
    original_module = sys.modules.pop(module_to_import, None)

    with patch.dict("sys.modules", modules):
        msg = (
            "`HuggingFacePubMedQA` requires the `huggingface-evals` extra to be installed. "
            "To fix please run `pip install fed-rag[huggingface-evals]`."
        )
        with pytest.raises(
            MissingExtraError,
            match=re.escape(msg),
        ):
            import fed_rag.evals.benchmarks as benchmarks

            benchmarks.HuggingFacePubMedQA()

    # restore module so to not affect other tests
    if original_module:
        sys.modules[module_to_import] = original_module
