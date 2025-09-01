"""Data structures for results"""

from typing import Any

from pydantic import BaseModel, Field


class TrainResult(BaseModel):
    """
    Represents the result of a training process.

    This class encapsulates the outcome of a model's training process,
    specifically storing the loss value calculated during training.

    Attributes:
    loss (float): The training loss value.
    """

    loss: float


class TestResult(BaseModel):
    """
    Represents the results of a test process, including loss and additional metrics.

    This class is used to encapsulate the results of testing, such as the calculated
    loss value and optional additional metrics. It includes fields for storing the
    primary loss value and a dictionary of computed metrics for more detailed analysis
    or performance evaluation. This ensures a structured representation of test outcomes.

    Attributes:
        loss (float): The primary loss value resulting from the test process.
        metrics (dict[str, Any]): Additional metrics computed on the test set. These can
            include various performance indicators or statistics relevant to the test.
    """

    __test__ = (
        False  # needed for Pytest collision. Avoids PytestCollectionWarning
    )
    loss: float
    metrics: dict[str, Any] = Field(
        description="Additional metrics computed on test set.",
        default_factory=dict,
    )
