"""CommonsenseQA

Example
===
{
 'question_id': 'Q1',
 'question': 'how are glacier caves formed?	',
 'document_title': 'Glacier cave',
 'answer': 'A partly submerged glacier cave on Perito Moreno Glacier .'
 'label': 0
},
{
 'question_id': 'Q1',
 'question': 'how are glacier caves formed?	',
 'document_title': 'Glacier cave',
 'answer': 'A glacier cave is a cave formed within the ice of a glacier .'
 'label': 1
}
"""

import numpy as np
import pandas as pd

from ..base_data_prepper import DEFAULT_SAVE_DIR, BaseDataPrepper
from .mixin import QAMixin

QA_SAVE_DIR = DEFAULT_SAVE_DIR / "qa"


class WikiQADataPrepper(QAMixin, BaseDataPrepper):
    @property
    def dataset_name(self) -> str:
        return "wiki_qa"

    def _get_answer(self, row: pd.Series) -> str:
        return str(row["answer"])

    def _prep_df(self) -> None:
        self.df["answer"] = self.df.apply(
            lambda row: self._get_answer(row), axis=1
        )

    def example_to_json(self, row: pd.Series) -> dict[str, str]:
        instruction_example: WikiQADataPrepper.InstructionExample = {
            "answer": row["answer"],
            "question": row["question"],
            "evidence": None,
        }
        return instruction_example  # type:ignore [return-value]


splits = {
    'test': 'data/test-00000-of-00001.parquet', 
    'validation': 'data/validation-00000-of-00001.parquet', 
    'train': 'data/train-00000-of-00001.parquet'
}

df = pd.read_parquet("hf://datasets/microsoft/wiki_qa/" + splits["test"])
df = df[df["label"] == 1]
data_prepper = WikiQADataPrepper(df=df, save_dir=QA_SAVE_DIR)
data_prepper.execute_and_save()
