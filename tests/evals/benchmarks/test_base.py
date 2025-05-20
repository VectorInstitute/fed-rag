from . import _benchmarks as benchmarks


def test_sequence_interface() -> None:
    # typical pattern
    test_benchmark = benchmarks.TestBenchmark()

    assert len(test_benchmark) == 3
    assert test_benchmark.num_examples == 3
    for ix in range(len(test_benchmark)):
        assert test_benchmark[ix] == test_benchmark._examples[ix]
    example_iter = iter(test_benchmark.as_iterator())
    assert next(example_iter) == test_benchmark[0]
