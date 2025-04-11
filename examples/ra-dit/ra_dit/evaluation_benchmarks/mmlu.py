"""MMLU Benchmark."""

import pandas as pd
from pydantic import PrivateAttr

from fed_rag.generators.hf_peft_model import HFPeftModelGenerator
from fed_rag.types.rag_system import RAGSystem

from .base import BaseBenchmark, ExamplePred, ScoredExamplePred

GENERATE_TEMPLATE = """
<role>
You are a helpful assistant.
</role>

<instruction>
You are given a question along with 4 choices as a potential answer. Additionally,
you are given some background context that may or may not be helpful. Respond
with the best choice to the question.
</instruction>

<warning>
- Only answer with the a single letter: "A", "B", "C", "D" representing your choice
</warning>

<question>
{question}
</question>

<background-context>
{context}
</background-context>

<response>
"""


class MMLUBenchmark(BaseBenchmark):
    generate_prompt_template: str | None = GENERATE_TEMPLATE
    _class_labels: dict[int, str] = PrivateAttr(
        default={0: "A", 1: "B", 2: "C", 3: "D"}
    )

    @property
    def name(self) -> str:
        return "MMLU"

    def _format_question(self, example: pd.Series) -> str:
        return str(
            example["question"]
            + "\n\n<choices>\n"
            + "\n".join(
                f"{choice_id}: {choice}"
                for choice_id, choice in zip(
                    ["A", "B", "C", "D"], example["choices"]
                )
            )
            + "\n</choices>"
        )

    def _predict_example(
        self, example: pd.Series, rag_system: RAGSystem
    ) -> ExamplePred:
        query = self._format_question(example)
        self.logger.debug(f"_predict_example created query:\n{query}")
        response = rag_system.query(query)

        return ExamplePred(pred=response.response)

    def _evaluate_prediction(
        self, example: pd.Series, pred: ExamplePred
    ) -> ScoredExamplePred:
        score = int(pred.pred == self._class_labels[example["answer"]])
        return ScoredExamplePred.from_example_pred(pred=pred, score=score)

    def _aggregate_example_scores(
        self, score_examples: list[ScoredExamplePred]
    ) -> float:
        return sum(ex.score for ex in score_examples) / len(score_examples)


splits = {
    "test": "abstract_algebra/test-00000-of-00001.parquet",
    "validation": "abstract_algebra/validation-00000-of-00001.parquet",
    "dev": "abstract_algebra/dev-00000-of-00001.parquet",
}
df = pd.read_parquet("hf://datasets/cais/mmlu/" + splits["test"])
mmlu_benchmark = MMLUBenchmark(examples=df.head(n=5))


if __name__ == "__main__":
    from ..rag_system import main as get_rag_system

    # example
    ix = 0
    example = mmlu_benchmark.examples.iloc[ix]

    # rag system
    rag_system = get_rag_system(
        retriever_id="dragon",
        generator_id="llama2_7b",
        generator_variant="qlora",
    )

    if isinstance(rag_system.generator, HFPeftModelGenerator):
        rag_system.generator.model = (
            rag_system.generator.model.merge_and_unload()
        )
    pred = mmlu_benchmark._predict_example(
        example=example, rag_system=rag_system
    )
    print(pred)
