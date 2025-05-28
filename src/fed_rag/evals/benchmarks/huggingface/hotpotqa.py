"""HotpotQA benchmark"""

from typing import Any

from pydantic import model_validator

from fed_rag.base.evals.benchmark import BaseBenchmark

from .mixin import HuggingFaceBenchmarkMixin
from .utils import check_huggingface_evals_installed


class HuggingFaceHotpotQA(HuggingFaceBenchmarkMixin, BaseBenchmark):
    """HuggingFace HotpotQA Benchmark.

    HotpotQA is a multi-hop question answering dataset that requires reasoning
    over multiple paragraphs to answer questions.

    Example schema:
        {
            "id": "5a7a06935542990198eaf050",
            "question": "Which magazine was started first Arthur's Magazine or First for Women?",
            "answer": "Arthur's Magazine",
            "type": "comparison",
            "level": "medium",
            "supporting_facts": {
                "title": ["Arthur's Magazine", "First for Women"],
                "sent_id": [0, 0]
            },
            "context": {
                "title": [
                    "Radio City (Indian radio station)",
                    "History of Albanian football",
                    "Echosmith",
                    "Women's colleges in the Southern United States",
                    "First Arthur County Courthouse and Jail",
                    "Arthur's Magazine",
                    "2012–13 Ukrainian Hockey Championship",
                    "First for Women",
                    "Freeway Complex Fire",
                    "William Rast"
                ],
                "sentences": [
                    ["Radio City is India's first private FM...", "It plays Hindi..."],
                    ["Football in Albania existed before...", "The Albanian..."],
                    ...
                    ["Arthur's Magazine (1844–1846) was an American literary periodical...", "It was founded by..."],
                    ...
                    ["First for Women is a woman's magazine published by...", "The magazine was started in 1989."],
                    ...
                ]
            }
        }
    """

    dataset_name = "hotpot_qa/hotpot_qa"
    configuration_name: str = "distractor"

    def _get_query_from_example(self, example: dict[str, Any]) -> str:
        return str(example["question"])

    def _get_response_from_example(self, example: dict[str, Any]) -> str:
        return str(example["answer"])

    def _get_context_from_example(self, example: dict[str, Any]) -> str | None:
        context = example.get("context", {})
        if isinstance(context, dict):
            titles = context.get("title", [])
            sentences = context.get("sentences", [])

            # Build context by combining titles with their sentences
            context_parts = []
            for i, (title, sents) in enumerate(zip(titles, sentences)):
                if isinstance(sents, list):
                    context_parts.append(f"{title}: {' '.join(sents)}")
                else:
                    context_parts.append(f"{title}: {sents}")

            return " ".join(context_parts) if context_parts else None
        else:
            # If context is not a dict, return None
            return None

    @model_validator(mode="before")
    @classmethod
    def _validate_extra_installed(cls, data: Any) -> Any:
        """Validate that huggingface-evals dependencies are installed."""
        check_huggingface_evals_installed(cls.__name__)
        return data
