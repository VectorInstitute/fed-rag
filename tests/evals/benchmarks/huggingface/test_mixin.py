from unittest.mock import MagicMock, patch

import pytest
from datasets import (
    Dataset,
    DatasetInfo,
    IterableDataset,
    SplitDict,
    SplitInfo,
)

from fed_rag.data_structures.evals import BenchmarkExample
from fed_rag.exceptions import EvalsError

from .. import _benchmarks as benchmarks


@patch("datasets.load_dataset")
def test_hf_mixin(
    mock_load_dataset: MagicMock, dummy_dataset: Dataset
) -> None:
    mock_load_dataset.return_value = dummy_dataset
    test_hf_benchmark = benchmarks.TestHFBenchmark()

    assert test_hf_benchmark.num_examples == 3
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
    mock_load_dataset: MagicMock,
) -> None:
    mock_load_dataset.side_effect = RuntimeError("dataset load fail")

    with pytest.raises(
        EvalsError,
        match="Failed to load dataset, `test_benchmark`, due to error: dataset load fail",
    ):
        benchmarks.TestHFBenchmark()


@patch("datasets.load_dataset")
def test_hf_mixin_raises_error_if_num_examples_if_no_listed_splits(
    mock_load_dataset: MagicMock, dummy_dataset: Dataset
) -> None:
    mock_load_dataset.return_value = dummy_dataset
    test_hf_benchmark = benchmarks.TestHFBenchmark()

    msg = (
        f"Unable to get size of dataset: `{test_hf_benchmark.dataset_name}`. "
        "The dataset does not have any listed splits."
    )
    with pytest.raises(EvalsError, match=msg):
        new_info = DatasetInfo(splits=None)
        test_hf_benchmark.dataset.info.update(new_info, ignore_none=False)

        print(test_hf_benchmark.dataset.info)

        # try to get this property but this should fail
        test_hf_benchmark.num_examples


@patch("datasets.load_dataset")
def test_hf_mixin_raises_error_if_num_examples_if_splits_does_not_contain_test(
    mock_load_dataset: MagicMock, dummy_dataset: Dataset
) -> None:
    mock_load_dataset.return_value = dummy_dataset
    test_hf_benchmark = benchmarks.TestHFBenchmark()

    msg = (
        f"Unable to get size of dataset: `{test_hf_benchmark.dataset_name}`. "
        f"Split, `{test_hf_benchmark.split}` does not exist in the splits of the dataset."
    )
    with pytest.raises(EvalsError, match=msg):
        split_dict = SplitDict()
        split_dict.add(SplitInfo(name="train", num_examples=3))
        new_info = DatasetInfo(splits=split_dict)
        test_hf_benchmark.dataset.info.update(new_info, ignore_none=False)

        print(test_hf_benchmark.dataset.info)

        # try to get this property but this should fail
        test_hf_benchmark.num_examples
