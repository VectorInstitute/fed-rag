"""Common abstractions for inspectors"""

from pydantic import BaseModel


class TrainerSignatureSpec(BaseModel):
    net_parameter: str
    train_data_param: str
    val_data_param: str
    extra_train_kwargs: list[str] = []
    net_parameter_class_name: str


class TesterSignatureSpec(BaseModel):
    __test__ = (
        False  # needed for Pytest collision. Avoids PytestCollectionWarning
    )
    net_parameter: str
    test_data_param: str
    extra_test_kwargs: list[str] = []
    net_parameter_class_name: str
