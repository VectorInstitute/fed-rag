"""QA Data Prepper"""

from abc import ABC, abstractmethod
from typing import TypedDict

import pandas as pd


class QAMixin(ABC):
    @property
    def required_cols(self) -> list[str]:
        return ["answer", "question"]

    @abstractmethod
    def _get_answer(self, row: pd.Series) -> str:
        """Get answer from an example row."""

    @abstractmethod
    def _get_question(self, row: pd.Series) -> str:
        """Get question from an example row."""

    @abstractmethod
    def _get_evidence(self, row: pd.Series) -> str | None:
        """Get evidence from an example row."""

    def _prep_df(self) -> None:
        if not hasattr(self, "df"):
            raise ValueError("Missing 'df' property.")
        if not isinstance(self.df, pd.DataFrame):
            raise ValueError(
                "Invalid type for 'df' property. Should be a ~pd.DataFrame."
            )
        self.df["answer"] = self.df.apply(
            lambda row: self._get_answer(row), axis=1
        )
        self.df["question"] = self.df.apply(
            lambda row: self._get_question(row), axis=1
        )
        self.df["evidence"] = self.df.apply(
            lambda row: self._get_evidence(row), axis=1
        )

    class InstructionExample(TypedDict):
        answer: str
        question: str
        evidence: str | None

    def example_to_json(self, row: pd.Series) -> dict[str, str]:
        instruction_example: QAMixin.InstructionExample = {
            "answer": row["answer"],
            "question": row["question"],
            "evidence": None,
        }
        return instruction_example  # type:ignore [return-value]
