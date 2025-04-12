"""PubmedQA

Example
===
{
    "question": ...,
    "context": {
            "contexts": [],
            ...
        },
    "long_answer": ...,
    "final_decision": ...
}
"""

import pandas as pd

from ..base_data_prepper import DEFAULT_SAVE_DIR, BaseDataPrepper
from .mixin import QAMixin

QA_SAVE_DIR = DEFAULT_SAVE_DIR / "qa"


class PubmedQADataPrepper(QAMixin, BaseDataPrepper):
    @property
    def dataset_name(self) -> str:
        return "pubmed_qa"

    def _get_answer(self, row: pd.Series) -> str:
        return str(row["long_answer"] + "\n\n" + row["final_decision"])

    def _get_evidence(self, row: pd.Series) -> str:
        return "\n\n".join(row["context"]["contexts"])

    def _get_question(self, row: pd.Series) -> str:
        return str(row["question"])


df = pd.read_parquet(
    "hf://datasets/qiaojin/PubMedQA/pqa_artificial/train-00000-of-00001.parquet"
)
data_prepper = PubmedQADataPrepper(df=df, save_dir=QA_SAVE_DIR)
data_prepper.execute_and_save()
