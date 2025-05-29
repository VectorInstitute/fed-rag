"""Natural Questions benchmark"""

from typing import Any

from pydantic import model_validator

from fed_rag.base.evals.benchmark import BaseBenchmark

from .mixin import HuggingFaceBenchmarkMixin
from .utils import check_huggingface_evals_installed


class HuggingFaceNaturalQuestions(HuggingFaceBenchmarkMixin, BaseBenchmark):
    """HuggingFace Natural Questions Benchmark.

    Natural Questions is a question answering dataset containing real user
    questions issued to Google search, paired with Wikipedia articles containing
    the answer. Each question may have a long answer (a paragraph) and/or a
    short answer (a span within the paragraph).

    Example schema:
        {
            "id": "5225754983651766092",
            "document": {
                "title": "Trade wind",
                "url": "https://en.wikipedia.org/wiki/Trade_wind",
                "html": "<!DOCTYPE html>...",
                "tokens": {
                    "token": ["Trade", "winds", "are", "the", "pattern", "of", "easterly", "surface", "winds", ...],
                    "is_html": [False, False, False, False, False, False, False, False, False, ...],
                    "start_byte": [0, 6, 12, 16, 20, 28, 30, 38, 46, ...],
                    "end_byte": [5, 11, 15, 19, 27, 29, 37, 45, 51, ...]
                }
            },
            "question": {
                "text": "what purpose did seasonal monsoon winds have on trade",
                "tokens": ["what", "purpose", "did", "seasonal", "monsoon", "winds", "have", "on", "trade"]
            },
            "long_answer_candidates": {
                "start_byte": [43178, 44667, 48014, 49619, 50365, 52946, 54877, 56067, 57202, 58657],
                "end_byte": [44666, 45901, 49618, 50364, 50993, 54876, 56066, 56879, 58656, 60294],
                "start_token": [44, 161, 343, 547, 628, 749, 949, 1046, 1141, 1304],
                "end_token": [161, 285, 547, 628, 701, 949, 1046, 1134, 1304, 1488],
                "top_level": [True, True, True, True, True, True, True, True, True, True]
            },
            "annotations": {
                "id": ["4323936797498927989", "13037645000009169623", "4439059471919323171", "15051359051424338858", "5332861748513580580"],
                "long_answer": [
                    {"candidate_index": -1, "start_byte": -1, "end_byte": -1, "start_token": -1, "end_token": -1},
                    {"candidate_index": -1, "start_byte": -1, "end_byte": -1, "start_token": -1, "end_token": -1},
                    {"candidate_index": -1, "start_byte": -1, "end_byte": -1, "start_token": -1, "end_token": -1},
                    {"candidate_index": 0, "start_byte": 43178, "end_byte": 44666, "start_token": 44, "end_token": 161},
                    {"candidate_index": 0, "start_byte": 43178, "end_byte": 44666, "start_token": 44, "end_token": 161}
                ],
                "short_answers": [
                    {"start_byte": [], "end_byte": [], "start_token": [], "end_token": [], "text": []},
                    {"start_byte": [], "end_byte": [], "start_token": [], "end_token": [], "text": []},
                    {"start_byte": [], "end_byte": [], "start_token": [], "end_token": [], "text": []},
                    {"start_byte": [44318], "end_byte": [44657], "start_token": [140], "end_token": [159],
                     "text": ["enabled European empire expansion into the Americas and trade routes to become established across the Atlantic and Pacific oceans"]},
                    {"start_byte": [], "end_byte": [], "start_token": [], "end_token": [], "text": []}
                ],
                "yes_no_answer": [-1, -1, -1, -1, -1]
            }
        }
    """

    dataset_name = "google-research-datasets/natural_questions"
    configuration_name: str = "default"
    split: str = (
        "validation"  # Natural Questions uses 'validation' instead of 'test'
    )

    def _get_query_from_example(self, example: dict[str, Any]) -> str:
        question = example.get("question", {})
        return str(question.get("text", ""))

    def _get_response_from_example(self, example: dict[str, Any]) -> str:
        annotations = example.get("annotations", {})

        # Get the lists from annotations
        yes_no_answers = annotations.get("yes_no_answer", [])
        short_answers_list = annotations.get("short_answers", [])
        long_answers_list = annotations.get("long_answer", [])

        if (
            not yes_no_answers
            and not short_answers_list
            and not long_answers_list
        ):
            return "[NO ANSWER]"

        # Check each annotation for valid answers
        for i, yes_no_answer in enumerate(yes_no_answers):
            # Check for yes/no answer first (-1 means no yes/no answer)
            if yes_no_answer == 1:
                return "YES"
            elif yes_no_answer == 0:
                return "NO"

            # Check for short answers
            if i < len(short_answers_list):
                short_answer = short_answers_list[i]
                texts = short_answer.get("text", [])
                if texts:  # If there are short answer texts
                    # Join multiple short answers with "and"
                    return " and ".join(str(text) for text in texts if text)

            # Check for long answers
            if i < len(long_answers_list):
                long_answer = long_answers_list[i]
                if long_answer.get("candidate_index", -1) >= 0:
                    return "[LONG ANSWER EXISTS]"

        return "[NO ANSWER]"

    def _get_context_from_example(self, example: dict[str, Any]) -> str | None:
        document = example.get("document", {})

        # Extract clean text from tokens (excluding HTML tags)
        tokens = document.get("tokens", {})
        token_list = tokens.get("token", [])
        is_html_list = tokens.get("is_html", [])

        if token_list and is_html_list:
            # Filter out HTML tokens and join text tokens
            text_tokens = [
                token
                for token, is_html in zip(token_list, is_html_list)
                if not is_html
            ]
            if text_tokens:
                return " ".join(text_tokens)

        # Fallback to title as minimal context
        title = document.get("title", "")
        return title if title else None

    @model_validator(mode="before")
    @classmethod
    def _validate_extra_installed(cls, data: Any) -> Any:
        """Validate that huggingface-evals dependencies are installed."""
        check_huggingface_evals_installed(cls.__name__)
        return data
