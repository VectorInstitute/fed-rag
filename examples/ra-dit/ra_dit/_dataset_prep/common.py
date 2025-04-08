"""Base Data Prepper Class"""

from abc import ABC, abstractmethod

import pandas as pd
from pydantic import BaseModel, Field


class BaseDataPrepper(ABC, BaseModel):
    df: pd.DataFrame = Field(description="Underlying Dataframe.")

    @property
    @abstractmethod
    def required_cols(self) -> list[str]:
        """The required cols in the attached df to create an instruction json."""

    @abstractmethod
    def prep_df(self) -> None:
        """Prepare df with required keys."""

    @abstractmethod
    def example_to_json(self, row: pd.Series) -> dict[str, str]:
        """Convert an example/row from the attached to df to an instruction json."""

    def to_jsons(self) -> list[dict[str, str]]:
        if self.required_cols not in self.df.columns:
            raise ValueError(
                f"The required cols: {self.required_cols} are not found in the attached df."
            )
        examples = []
        for _, row in self.df.iterrows():
            examples.append(self.example_to_json(row))
        return examples
