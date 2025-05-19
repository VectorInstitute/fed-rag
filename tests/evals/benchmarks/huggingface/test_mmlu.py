from unittest.mock import MagicMock, patch

import fed_rag.evals.benchmarks as benchmarks


@patch("datasets.load_dataset")
def test_hf_mixin(mock_load_dataset: MagicMock) -> None:
    # arrange
    test_hf_benchmark = benchmarks.HuggingFaceMMLU()

    mock_load_dataset.assert_called_once_with(
        benchmarks.HuggingFaceMMLU.dataset_name,
        name=test_hf_benchmark.configuration_name,
        split=test_hf_benchmark.split,
        streaming=test_hf_benchmark.streaming,
    )
