"""Training Args"""

from typing import Any

from pydantic import BaseModel, Field


class TrainingArgs(BaseModel):
    """Arguments for training."""

    learning_rate: float | None = None
    batch_size: int | None = None
    num_epochs: int | None = None
    warmup_steps: int | None = None
    weight_decay: float | None = None
    custom_kwargs: dict[str, Any] = Field(default_factory=dict)
