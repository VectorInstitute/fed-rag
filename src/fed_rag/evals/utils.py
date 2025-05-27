"""Utils module for evals"""

import json
from pathlib import Path

from fed_rag.data_structures.evals import BenchmarkEvaluatedExample


def load_evaluations(filename: str | Path) -> list[BenchmarkEvaluatedExample]:
    """Utility for loading serialized BenchmarkEvaluatedExamples in a JSONL file."""

    if isinstance(filename, str):
        filename = Path(filename)

    if not filename.exists():
        raise FileNotFoundError(f"Evaluation file not found: {filename}")

    with open(filename, "r") as f:
        data = [json.loads(line) for line in f]

    return [BenchmarkEvaluatedExample(**item) for item in data]
