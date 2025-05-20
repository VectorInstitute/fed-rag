from unittest.mock import MagicMock, patch

import pytest
from datasets import Dataset, IterableDataset

from fed_rag.data_structures.evals import BenchmarkExample
from fed_rag.exceptions import EvalsError

from .. import _benchmarks as benchmarks


@patch("datasets.load_dataset")
def test_hf_mixin(
    mock_load_dataset: MagicMock, dummy_dataset: Dataset
) -> None:
    mock_load_dataset.return_value = dummy_dataset
    test_hf_benchmark = benchmarks.TestHFBenchmark()

    assert len(test_hf_benchmark) == 3
    assert (
        test_hf_benchmark.dataset_name
        == test_hf_benchmark._dataset.info.dataset_name
    )
    assert isinstance(test_hf_benchmark[0], BenchmarkExample)


@patch("datasets.load_dataset")
def test_hf_streaming(
    mock_load_dataset: MagicMock, dummy_iterable_dataset: IterableDataset
) -> None:
    mock_load_dataset.return_value = dummy_iterable_dataset
    test_hf_benchmark = benchmarks.TestHFBenchmark(streaming=True)

    assert isinstance(test_hf_benchmark.dataset, IterableDataset)

    example_stream = test_hf_benchmark.as_stream()
    next(example_stream)
    next(example_stream)
    next(example_stream)
    with pytest.raises(StopIteration):
        next(example_stream)


@patch("datasets.load_dataset")
def test_hf_convert_to_streaming(
    mock_load_dataset: MagicMock,
    dummy_dataset: Dataset,
    dummy_iterable_dataset: IterableDataset,
) -> None:
    mock_load_dataset.return_value = dummy_dataset
    mock_to_iterable_dataset = MagicMock()
    mock_to_iterable_dataset.return_value = dummy_iterable_dataset
    dummy_dataset.to_iterable_dataset = mock_to_iterable_dataset
    test_hf_benchmark = benchmarks.TestHFBenchmark()

    assert isinstance(test_hf_benchmark.dataset, Dataset)

    example_stream = test_hf_benchmark.as_stream()
    next(example_stream)
    next(example_stream)
    next(example_stream)
    with pytest.raises(StopIteration):
        next(example_stream)


@patch("datasets.load_dataset")
def test_hf_mixin_raises_error_if_load_dataset_fails(
    mock_load_dataset: MagicMock, dummy_dataset: Dataset
) -> None:
    mock_load_dataset.side_effect = RuntimeError("dataset load fail")

    with pytest.raises(
        EvalsError,
        match="Failed to load dataset, `test_benchmark`, due to error: dataset load fail",
    ):
        benchmarks.TestHFBenchmark()
