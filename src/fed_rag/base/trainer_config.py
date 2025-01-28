"""Base Trainer Config"""

from typing import Any, Dict, cast

from pydantic import BaseModel, ConfigDict, PrivateAttr, model_serializer


class BaseTrainerConfig(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    net: Any
    train_data: Any
    val_data: Any
    _extra_train_kwargs: Dict[str, Any] = PrivateAttr(
        default_factory=dict
    )  # additional kwargs

    def __init__(self, **params: Any):
        """__init__.

        Sets specified fields and private attrs of the TrainerConfig and then
        stores any additional passed params in _extra_train_kwargs.
        """
        fields = {}
        private_attrs = {}
        extra_train_kwargs = {}
        for k, v in params.items():
            if k in self.model_fields:
                fields[k] = v
            elif k in self.__private_attributes__:
                private_attrs[k] = v
            else:
                extra_train_kwargs[k] = v
        super().__init__(**fields)
        for private_attr, value in private_attrs.items():
            super().__setattr__(private_attr, value)
        if extra_train_kwargs:
            self._extra_train_kwargs.update(extra_train_kwargs)

    @model_serializer(mode="wrap")
    def custom_model_dump(self, handler: Any) -> Dict[str, Any]:
        data = handler(self)
        data = cast(Dict[str, Any], data)
        # include _extra_train_kwargs in serialization
        if self._extra_train_kwargs:
            data["_extra_train_kwargs"] = self._extra_train_kwargs
        return data  # type: ignore[no-any-return]
