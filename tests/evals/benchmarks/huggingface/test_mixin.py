from unittest.mock import MagicMock, patch

from datasets import Dataset

from fed_rag.data_structures.evals import BenchmarkExample

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
    mock_load_dataset: MagicMock, dummy_dataset: Dataset
) -> None:
    mock_load_dataset.return_value = dummy_dataset
    test_hf_benchmark = benchmarks.TestHFBenchmark(streaming=True)

    example_stream = test_hf_benchmark.as_stream()
    print(next(example_stream).model_dump())
