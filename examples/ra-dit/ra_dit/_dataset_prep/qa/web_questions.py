"""WebQA

Example
===
{
 'url': 'http://www.freebase.com/view/en/justin_bieber',
 'question': 'http://www.freebase.com/view/en/justin_bieber',
 'answer': 'answers'
},
"""

import pandas as pd

from ..base_data_prepper import DEFAULT_SAVE_DIR, BaseDataPrepper
from .mixin import QAMixin

QA_SAVE_DIR = DEFAULT_SAVE_DIR / "qa"


class WebQuestionsDataPrepper(QAMixin, BaseDataPrepper):
    @property
    def dataset_name(self) -> str:
        return "web_questions_qa"

    def _get_answer(self, row: pd.Series) -> str:
        return str(", ".join(row["answers"]))

    def _get_evidence(self, row: pd.Series) -> str | None:
        return None

    def _get_question(self, row: pd.Series) -> str:
        return str(row["question"])


splits = {
    "train": "data/train-00000-of-00001.parquet",
    "test": "data/test-00000-of-00001.parquet",
}

df = pd.read_parquet("hf://datasets/Stanford/web_questions/" + splits["train"])
data_prepper = WebQuestionsDataPrepper(df=df, save_dir=QA_SAVE_DIR)
data_prepper.execute_and_save()
