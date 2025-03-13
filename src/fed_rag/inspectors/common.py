"""Common abstractions for inspectors"""

from pydantic import BaseModel


class TrainerSignatureSpec(BaseModel):
    net_parameter: str
    train_data_param: str
    val_data_param: str
    extra_train_kwargs: list[str] = []
