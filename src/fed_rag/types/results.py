"""TesterResult"""

from typing import Any

from pydantic import BaseModel, Field


class TrainResult(BaseModel):
    loss: float


class TestResult(BaseModel):
    loss: float
    metrics: dict[str, Any] = Field(
        description="Additional metrics computed on test set.",
        default_factory=dict,
    )
