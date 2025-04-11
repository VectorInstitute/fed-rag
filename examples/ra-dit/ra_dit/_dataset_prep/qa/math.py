"""CommonsenseQA

Example
===
{
 'Problem': 'the banker ' s gain of a certain sum due 3 yea...',
 'Rationale': 'explanation : t = 3 years r = 10 % td = ( bg ...',
 'options': 'a ) rs . 400 , b ) rs . 300 , c ) rs . 500 , d...',
 'correct': 'a',
 'annotated_formula': 'divide(multiply(const_100, divide(multiply(36,...',
 'linear_formula': 'multiply(n2,const_100)|multiply(n0,n1)|divide(...	',
 'category' : 'gain'
}
"""

import re

import pandas as pd
from datasets import load_dataset

from ..base_data_prepper import DEFAULT_SAVE_DIR, BaseDataPrepper
from .mixin import QAMixin

QA_SAVE_DIR = DEFAULT_SAVE_DIR / "qa"


class MathQADataPrepper(QAMixin, BaseDataPrepper):
    @property
    def dataset_name(self) -> str:
        return "math_qa"

    def _get_answer(self, row: pd.Series) -> str:
        options = re.findall(r"([a-z])\s*\)\s*([^,]+)", row["answer"])
        for label, text in options:
            if label.strip() == row["correct"].strip():
                answer = text.strip()
        return str(answer)

    def _prep_df(self) -> None:
        self.df["answer"] = self.df.apply(
            lambda row: self._get_answer(row), axis=1
        )

    def example_to_json(self, row: pd.Series) -> dict[str, str]:
        instruction_example: MathQADataPrepper.InstructionExample = {
            "answer": row["answer"],
            "question": row["question"],
            "evidence": None,
        }
        return instruction_example  # type:ignore [return-value]


splits = {"test": "test", "validation": "valid", "train": "train"}

dataset = load_dataset("allenai/math_qa", trust_remote_code=True)
df = pd.DataFrame(dataset["train"])
df = df.rename(columns={"Problem": "question", "options": "answer"})
data_prepper = MathQADataPrepper(df=df, save_dir=QA_SAVE_DIR)
data_prepper.execute_and_save()
