"""QA Data Prepper"""

from typing import TypedDict


class QAMixin:
    @property
    def required_cols(self) -> list[str]:
        return ["answer", "question"]

    class InstructionExample(TypedDict):
        answer: str
        question: str
        evidence: str | None
