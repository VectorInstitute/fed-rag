"""CommonsenseQA

Example
===
{'id': '075e483d21c29a511267ef62bedc0461',
 'question': 'The sanctions against the school were a punishing blow, and they seemed to what the efforts the school had made to change?',
 'question_concept': 'punishing',
 'choices': {'label': ['A', 'B', 'C', 'D', 'E'],
  'text': ['ignore', 'enforce', 'authoritarian', 'yell at', 'avoid']},
 'answerKey': 'A'}
"""

import numpy as np
import pandas as pd

from ..base_data_prepper import DEFAULT_SAVE_DIR, BaseDataPrepper
from .mixin import QAMixin

QA_SAVE_DIR = DEFAULT_SAVE_DIR / "qa"


class CommonsenseQADataPrepper(QAMixin, BaseDataPrepper):
    @property
    def dataset_name(self) -> str:
        return "commonsense_qa"

    def _get_answer(self, row: pd.Series) -> str:
        answer_ix = np.where(row["choices"]["label"] == row["answerKey"])
        return str(row["choices"]["text"][answer_ix][0])

    def _get_question(self, row: pd.Series) -> str:
        return str(row["question"])

    def _get_evidence(self, row: pd.Series) -> str | None:
        return None


splits = {
    "train": "data/train-00000-of-00001.parquet",
    "validation": "data/validation-00000-of-00001.parquet",
    "test": "data/test-00000-of-00001.parquet",
}
df = pd.read_parquet("hf://datasets/tau/commonsense_qa/" + splits["train"])
data_prepper = CommonsenseQADataPrepper(df=df, save_dir=QA_SAVE_DIR)
data_prepper.execute_and_save()
