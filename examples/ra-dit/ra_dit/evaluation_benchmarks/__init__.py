from .base import BenchmarkResult
from .mmlu import mmlu_benchmark

benchmarks = {"mmlu": mmlu_benchmark}

__all__ = ["BenchmarkResult"]
