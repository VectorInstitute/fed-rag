from . import _benchmarks as benchmarks


def test_sequence_interface() -> None:
    # typical pattern
    my_benchmark = benchmarks.MyBenchmark()

    assert len(my_benchmark) == 3
    for ix in range(len(my_benchmark)):
        assert my_benchmark[ix] == my_benchmark._examples[ix]
    example_iter = iter(my_benchmark.as_iterator())
    assert next(example_iter) == my_benchmark[0]
