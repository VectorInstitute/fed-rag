"""Base Data Prepper Class"""

import json
import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import pandas as pd
import tqdm
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr
from ra_dit.logger import logger

DEFAULT_SAVE_DIR = Path(__file__).parents[2].absolute() / "data"


class BaseDataPrepper(BaseModel, ABC):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    df: pd.DataFrame = Field(description="Underlying Dataframe.")
    instruction_jsons: list[dict[str, str]] = Field(
        description="Instruction jsons.", default_factory=list
    )
    save_dir: Path
    _logger: logging.Logger = PrivateAttr()

    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self._logger = logging.getLogger(
            f"{logger.name}.{self.__class__.__name__}"
        )
        self._logger.info(f"Initializing {self.__class__.__name__}")

    @property
    @abstractmethod
    def dataset_name(self) -> str:
        """Name of dataset.

        Used for saving jsonl file.
        """

    @property
    def logger(self) -> logging.Logger:
        return self._logger

    @property
    @abstractmethod
    def required_cols(self) -> list[str]:
        """The required cols in the attached df to create an instruction json."""

    @abstractmethod
    def _prep_df(self) -> None:
        """Custom prep df logic. Subclasses implement this method."""

    def prep_df(self) -> None:
        """Prepare df with required keys."""
        self.logger.info("Starting data preparation")
        self._prep_df()
        self.logger.info("Completed data preparation")

    @abstractmethod
    def example_to_json(self, row: pd.Series) -> dict[str, str]:
        """Convert an example/row from the attached to df to an instruction json."""

    def populate_instruction_jsons(self) -> None:
        self.logger.info("Starting creation of instruction jsons")
        if not all(col in self.df.columns for col in self.required_cols):
            raise ValueError(
                f"The required cols: {self.required_cols} are not found in the attached df."
            )
        examples = []
        total_rows = len(self.df)
        log_interval = max(
            1, total_rows // 10
        )  # Log at least at start and end
        for ix, row in tqdm.tqdm(self.df.iterrows()):
            examples.append(self.example_to_json(row))

            if (ix + 1) % log_interval == 0:
                self.logger.debug(
                    f"Processed {ix + 1} / {total_rows} examples"
                )

        self.logger.info("Completed creation of instruction jsons")
        self.logger.debug(f"Created {len(examples)} instruction examples")
        self.instruction_jsons = examples

    def save_instructions_to_jsonl_file(self) -> None:
        filename = self.save_dir / f"{self.dataset_name}.jsonl"
        self.logger.info(
            f"Saving {len(self.instruction_jsons)} instructions to: {filename}"
        )
        with open(filename, "w") as outfile:
            for entry in self.instruction_jsons:
                json.dump(entry, outfile)
                outfile.write("\n")
        self.logger.info(
            f"Successfully saved instruction jsons to: {filename}"
        )
