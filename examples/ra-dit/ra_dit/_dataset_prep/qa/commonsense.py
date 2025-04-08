"""CommonsenseQA

{'id': '075e483d21c29a511267ef62bedc0461',
 'question': 'The sanctions against the school were a punishing blow, and they seemed to what the efforts the school had made to change?',
 'question_concept': 'punishing',
 'choices': {'label': ['A', 'B', 'C', 'D', 'E'],
  'text': ['ignore', 'enforce', 'authoritarian', 'yell at', 'avoid']},
 'answerKey': 'A'}
"""

from typing import TypedDict

import numpy as np
import pandas as pd

from ..base_data_prepper import DEFAULT_SAVE_DIR, BaseDataPrepper
from .mixin import QAMixin

QA_SAVE_DIR = DEFAULT_SAVE_DIR / "qa"


class CommonsenseQADataPrepper(QAMixin, BaseDataPrepper):
    class InstructionExample(TypedDict):
        answer: str
        question: str

    @property
    def dataset_name(self) -> str:
        return "commonsense_qa"

    def _get_answer(self, row: pd.Series) -> str:
        answer_ix = np.where(row["choices"]["label"] == row["answerKey"])
        return str(row["choices"]["text"][answer_ix][0])

    def _prep_df(self) -> None:
        self.df["answer"] = self.df.apply(
            lambda row: self._get_answer(row), axis=1
        )

    def example_to_json(self, row: pd.Series) -> dict[str, str]:
        return self.InstructionExample(  # type:ignore [return-value]
            answer=row["answer"], question=row["question"]
        )


splits = {
    "train": "data/train-00000-of-00001.parquet",
    "validation": "data/validation-00000-of-00001.parquet",
    "test": "data/test-00000-of-00001.parquet",
}
df = pd.read_parquet("hf://datasets/tau/commonsense_qa/" + splits["train"])
data_prepper = CommonsenseQADataPrepper(df=df, save_dir=QA_SAVE_DIR)
data_prepper.prep_df()
data_prepper.populate_instruction_jsons()
data_prepper.save_instructions_to_jsonl_file()
