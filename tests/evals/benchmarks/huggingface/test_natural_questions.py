"""Tests for Natural Questions benchmark"""

import re
import sys
from unittest.mock import MagicMock, patch

import pytest
from datasets import Dataset

import fed_rag.evals.benchmarks as benchmarks
from fed_rag.data_structures.evals import BenchmarkExample
from fed_rag.exceptions import MissingExtraError


@pytest.fixture
def dummy_natural_questions_short_answer() -> Dataset:
    """Create a dummy Natural Questions dataset with short answer."""
    return Dataset.from_dict(
        {
            "id": ["5655493461695674962"],
            "document": [
                {
                    "title": "Parma Heights, Ohio",
                    "url": "https://en.wikipedia.org/wiki/Parma_Heights,_Ohio",
                    "tokens": {
                        "token": [
                            "Parma",
                            "Heights",
                            "is",
                            "a",
                            "city",
                            "in",
                            "Cuyahoga",
                            "County",
                            ",",
                            "Ohio",
                            ".",
                        ],
                        "is_html": [
                            False,
                            False,
                            False,
                            False,
                            False,
                            False,
                            False,
                            False,
                            False,
                            False,
                            False,
                        ],
                        "start_byte": [
                            0,
                            6,
                            14,
                            17,
                            19,
                            24,
                            27,
                            36,
                            42,
                            44,
                            48,
                        ],
                        "end_byte": [
                            5,
                            13,
                            16,
                            18,
                            23,
                            26,
                            35,
                            42,
                            43,
                            48,
                            49,
                        ],
                    },
                }
            ],
            "question": [
                {
                    "text": "what county is parma heights ohio in",
                    "tokens": [
                        "what",
                        "county",
                        "is",
                        "parma",
                        "heights",
                        "ohio",
                        "in",
                    ],
                }
            ],
            "long_answer_candidates": [
                {
                    "start_byte": [0],
                    "end_byte": [49],
                    "start_token": [0],
                    "end_token": [10],
                    "top_level": [True],
                }
            ],
            "annotations": [
                {
                    "id": ["4569326352834458124"],
                    "long_answer": [
                        {
                            "start_byte": 0,
                            "end_byte": 49,
                            "start_token": 0,
                            "end_token": 10,
                            "candidate_index": 0,
                        }
                    ],
                    "short_answers": [
                        {
                            "start_byte": [27],
                            "end_byte": [42],
                            "start_token": [6],
                            "end_token": [7],
                            "text": ["Cuyahoga County"],
                        }
                    ],
                    "yes_no_answer": [-1],
                }
            ],
        }
    )


@pytest.fixture
def dummy_natural_questions_yes_no() -> Dataset:
    """Create a dummy Natural Questions dataset with yes/no answer."""
    return Dataset.from_dict(
        {
            "id": ["1"],
            "document": [
                {
                    "title": "Test",
                    "tokens": {
                        "token": ["Test", "document", "text", "."],
                        "is_html": [False, False, False, False],
                        "start_byte": [0, 5, 14, 18],
                        "end_byte": [4, 13, 18, 19],
                    },
                }
            ],
            "question": [
                {
                    "text": "Is this a test?",
                    "tokens": ["is", "this", "a", "test"],
                }
            ],
            "annotations": [
                {
                    "id": ["123"],
                    "long_answer": [
                        {
                            "candidate_index": -1,
                            "start_byte": -1,
                            "end_byte": -1,
                            "start_token": -1,
                            "end_token": -1,
                        }
                    ],
                    "short_answers": [
                        {
                            "start_byte": [],
                            "end_byte": [],
                            "start_token": [],
                            "end_token": [],
                            "text": [],
                        }
                    ],
                    "yes_no_answer": [1],  # YES
                }
            ],
        }
    )


@pytest.fixture
def dummy_natural_questions_no_answer() -> Dataset:
    """Create a dummy Natural Questions dataset with no answer."""
    return Dataset.from_dict(
        {
            "id": ["2"],
            "document": [
                {
                    "title": "Test",
                    "tokens": {
                        "token": ["Test", "document", "."],
                        "is_html": [False, False, False],
                        "start_byte": [0, 5, 13],
                        "end_byte": [4, 13, 14],
                    },
                }
            ],
            "question": [
                {
                    "text": "What is unknown?",
                    "tokens": ["what", "is", "unknown"],
                }
            ],
            "annotations": [
                {
                    "id": ["456"],
                    "long_answer": [
                        {
                            "candidate_index": -1,
                            "start_byte": -1,
                            "end_byte": -1,
                            "start_token": -1,
                            "end_token": -1,
                        }
                    ],
                    "short_answers": [
                        {
                            "start_byte": [],
                            "end_byte": [],
                            "start_token": [],
                            "end_token": [],
                            "text": [],
                        }
                    ],
                    "yes_no_answer": [-1],  # No yes/no answer
                }
            ],
        }
    )


@pytest.fixture
def dummy_natural_questions_multiple_short_answers() -> Dataset:
    """Create a dummy Natural Questions dataset with multiple short answers."""
    return Dataset.from_dict(
        {
            "id": ["3"],
            "document": [
                {
                    "title": "Google Founders",
                    "tokens": {
                        "token": [
                            "Google",
                            "was",
                            "founded",
                            "by",
                            "Larry",
                            "Page",
                            "and",
                            "Sergey",
                            "Brin",
                            ".",
                        ],
                        "is_html": [
                            False,
                            False,
                            False,
                            False,
                            False,
                            False,
                            False,
                            False,
                            False,
                            False,
                        ],
                        "start_byte": [0, 7, 11, 19, 22, 28, 33, 37, 44, 48],
                        "end_byte": [6, 10, 18, 21, 27, 32, 36, 43, 48, 49],
                    },
                }
            ],
            "question": [
                {
                    "text": "who founded google",
                    "tokens": ["who", "founded", "google"],
                }
            ],
            "annotations": [
                {
                    "id": ["789"],
                    "long_answer": [
                        {
                            "candidate_index": -1,
                            "start_byte": -1,
                            "end_byte": -1,
                            "start_token": -1,
                            "end_token": -1,
                        }
                    ],
                    "short_answers": [
                        {
                            "start_byte": [22, 37],
                            "end_byte": [32, 48],
                            "start_token": [4, 7],
                            "end_token": [5, 8],
                            "text": ["Larry Page", "Sergey Brin"],
                        }
                    ],
                    "yes_no_answer": [-1],
                }
            ],
        }
    )


@patch("datasets.load_dataset")
def test_natural_questions_benchmark(mock_load_dataset: MagicMock) -> None:
    # arrange & act
    nq = benchmarks.HuggingFaceNaturalQuestions()

    # assert
    mock_load_dataset.assert_called_once_with(
        benchmarks.HuggingFaceNaturalQuestions.dataset_name,
        name=nq.configuration_name,
        split=nq.split,  # Should be 'validation'
        streaming=nq.streaming,
    )


@patch("datasets.load_dataset")
def test_natural_questions_short_answer(
    mock_load_dataset: MagicMock, dummy_natural_questions_short_answer: Dataset
) -> None:
    # arrange
    mock_load_dataset.return_value = dummy_natural_questions_short_answer
    nq = benchmarks.HuggingFaceNaturalQuestions()

    # assert
    assert isinstance(nq[0], BenchmarkExample)
    assert nq[0].query == "what county is parma heights ohio in"
    assert nq[0].response == "Cuyahoga County"
    assert (
        nq[0].context == "Parma Heights is a city in Cuyahoga County , Ohio ."
    )


@patch("datasets.load_dataset")
def test_natural_questions_multiple_short_answers(
    mock_load_dataset: MagicMock,
    dummy_natural_questions_multiple_short_answers: Dataset,
) -> None:
    # arrange
    mock_load_dataset.return_value = (
        dummy_natural_questions_multiple_short_answers
    )
    nq = benchmarks.HuggingFaceNaturalQuestions()

    # assert
    assert nq[0].response == "Larry Page and Sergey Brin"


@patch("datasets.load_dataset")
def test_natural_questions_yes_no_answer(
    mock_load_dataset: MagicMock, dummy_natural_questions_yes_no: Dataset
) -> None:
    # arrange
    mock_load_dataset.return_value = dummy_natural_questions_yes_no
    nq = benchmarks.HuggingFaceNaturalQuestions()

    # assert
    assert nq[0].response == "YES"


@patch("datasets.load_dataset")
def test_natural_questions_no_answer(
    mock_load_dataset: MagicMock, dummy_natural_questions_no_answer: Dataset
) -> None:
    # arrange
    mock_load_dataset.return_value = dummy_natural_questions_no_answer
    nq = benchmarks.HuggingFaceNaturalQuestions()

    # assert
    assert nq[0].response == "[NO ANSWER]"


@patch("datasets.load_dataset")
def test_natural_questions_long_answer_only(
    mock_load_dataset: MagicMock,
) -> None:
    """Test handling when only long answer exists."""
    dataset = Dataset.from_dict(
        {
            "id": ["4"],
            "document": [
                {
                    "title": "Test",
                    "tokens": {
                        "token": ["Long", "document", "text", "."],
                        "is_html": [False, False, False, False],
                        "start_byte": [0, 5, 14, 18],
                        "end_byte": [4, 13, 18, 19],
                    },
                }
            ],
            "question": [
                {
                    "text": "What is this about?",
                    "tokens": ["what", "is", "this", "about"],
                }
            ],
            "annotations": [
                {
                    "id": ["789"],
                    "long_answer": [
                        {
                            "candidate_index": 0,
                            "start_byte": 0,
                            "end_byte": 19,
                            "start_token": 0,
                            "end_token": 3,
                        }
                    ],  # Valid long answer
                    "short_answers": [
                        {
                            "start_byte": [],
                            "end_byte": [],
                            "start_token": [],
                            "end_token": [],
                            "text": [],
                        }
                    ],
                    "yes_no_answer": [-1],
                }
            ],
        }
    )
    mock_load_dataset.return_value = dataset
    nq = benchmarks.HuggingFaceNaturalQuestions()
    assert nq[0].response == "[LONG ANSWER EXISTS]"


@patch("datasets.load_dataset")
def test_natural_questions_no_annotations(
    mock_load_dataset: MagicMock,
) -> None:
    """Test handling when annotations are missing."""
    dataset = Dataset.from_dict(
        {
            "id": ["5"],
            "document": [{"title": "Test"}],  # No tokens
            "question": [{"text": "Test question?"}],
            "annotations": [{}],  # Empty annotations
        }
    )
    mock_load_dataset.return_value = dataset
    nq = benchmarks.HuggingFaceNaturalQuestions()
    assert nq[0].response == "[NO ANSWER]"
    assert nq[0].context == "Test"  # Falls back to title


@patch("datasets.load_dataset")
def test_natural_questions_missing_fields(
    mock_load_dataset: MagicMock,
) -> None:
    """Test handling when various fields are missing."""
    dataset = Dataset.from_dict(
        {
            "id": ["6"],
            "document": [{}],  # Empty document
            "question": [{}],  # Empty question
            "annotations": [{"yes_no_answer": [-1]}],
        }
    )
    mock_load_dataset.return_value = dataset
    nq = benchmarks.HuggingFaceNaturalQuestions()
    assert nq[0].query == ""
    assert nq[0].response == "[NO ANSWER]"
    assert nq[0].context is None


def test_huggingface_evals_extra_missing() -> None:
    """Test that proper error is raised when huggingface-evals extra is missing."""
    modules = {
        "datasets": None,
    }
    module_to_import = "fed_rag.evals.benchmarks"
    original_module = sys.modules.pop(module_to_import, None)

    with patch.dict("sys.modules", modules):
        msg = (
            "`HuggingFaceNaturalQuestions` requires the `huggingface-evals` extra to be installed. "
            "To fix please run `pip install fed-rag[huggingface-evals]`."
        )
        with pytest.raises(
            MissingExtraError,
            match=re.escape(msg),
        ):
            import fed_rag.evals.benchmarks as benchmarks

            benchmarks.HuggingFaceNaturalQuestions()

    # restore module so to not affect other tests
    if original_module:
        sys.modules[module_to_import] = original_module
