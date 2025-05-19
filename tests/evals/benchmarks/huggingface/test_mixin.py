from pathlib import Path

from .. import _benchmarks as benchmarks

TEST_CACHE_DIR = Path(__file__).parents[2].absolute() / "datasets"


def test_hf_mixin() -> None:
    test_hf_benchmark = benchmarks.TestHFBenchmark(
        load_kwargs={"cache_dir": TEST_CACHE_DIR}
    )

    assert len(test_hf_benchmark) == 3
    assert (
        test_hf_benchmark.dataset_name.replace("nerdai/", "")
        == test_hf_benchmark._dataset.info.dataset_name
    )
