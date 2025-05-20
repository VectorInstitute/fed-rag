# Benchmark a RAG System

In this quick start guide, we'll demonstrate how to leverage the `evals` module
within the `fed-rag` library to benchmark your `RAGSystem`. For conciseness, we
won't cover the detailed process of assembling a `RAGSystem` here—please refer to
our other quick start guides for comprehensive instructions on system assembly.

## The `Benchmarker` Class

Within the `evals` module is a core class called [`Benchmarker`](../../api_reference/evals/benchmarker.md).
It bears the responsibility of running a benchmark for your `RAGSystem`.

```py title="Creating a benchmarker"
from fed_rag.evals import Benchmarker

benchmarker = Benchmarker(rag_system=rag_system)  # (1)!
```

1. Your previously assembled `RAGSystem`

## Importing a `Benchmark` to Run

The `evals` module contains various benchmarks that can be used to evaluate a
`RAGSystem`. A FedRAG benchmark contains [`BenchmarkExamples`](../../api_reference/data_structures/evals.md)
that carry the query, response, and context for any given example.

Inspired by how datasets are imported from familiar libraries like `torchvision`,
the benchmarks can be imported as follows:

```py title="Importing the benchmarks module"
import fed_rag.evals.benchmarks as benchmarks
```

From here, we can choose to use any of the defined benchmarks! The snippet below
makes use of the `HuggingFaceMMLU` benchmark.

```py title="Using a supported benchmark"
mmlu = benchmarks.HuggingFaceMMLU(streaming=True)  # (1)!

# get the example stream
examples_stream = mmlu.as_stream()
print(next(examples_stream))  # will yield the next BenchmarkExample for MMLU
```

1. The HuggingFace benchmarks integration supports the underlying streaming mechanism of `~datasets.Dataset`.

!!! info
    Using a HuggingFace supported benchmark, requires the `huggingface-evals` extra.
    This can be installed via `pip-install fed-rag[huggingface-evals]`. Note that
    the more comprehensive `huggingface` extra also includes all necessary packages
    for `huggingface-evals`.

## Choosing your Evaluation Metric

To run a benchmark, you must also supply a [`EvaluationMetric`](../../api_reference/evals/index.md).
The code snippet below imports the [`ExactMatchEvaluationMetric`](../../api_reference/evals/metrics/exact_match.md).

```py title="Defining an evaluation metric"
from fed_rag.evals.metrics import ExactMatchEvaluationMetric

metric = ExactMatchEvaluationMetric()

# using the metric
metric(prediction="A", acutal="a")  # case in-sensitive returns 1.0
```

!!! info
    All subclasses of `BaseEvaluationMetric`, like `ExactMatchEvaluationMetric`
    are callable. We can see the signature of this method by using the help builtin
    i.e., `help(metric.__call__)`.

## Running the Benchmark

We now have all the elements in place in order to run the benchmark. To do so,
we invoke the `run()` method of the `Benchmarker` object, passing in the elements
we defined in previous sections.

```py title="Running the chose benchmark with specific metric"
result = benchmark.run(
    benchmark=mmlu,
    metric=metric,
    is_streaming=True,
    num_examples=3,  # (1)!
    agg="avg",  # (2)!
)

print(result)
```

1. (Optional) useful for rapid testing of your benchmark rig.
2. Can be 'avg', 'sum', 'max', 'min', see [`AggregationMode`](../../api_reference/data_structures/evals.md)

A successful run of a benchmark will result in a [`BenchmarkResult`](../../api_reference/data_structures/evals.md)
object that contains summary information about the benchmark including the final
aggregated score, the number of examples used, as well as the total number of examples
that the benchmark contains.
