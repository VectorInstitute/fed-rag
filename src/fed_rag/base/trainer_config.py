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

    def __getattr__(self, __name: str) -> Any:
        if (
            __name in self.__private_attributes__
            or __name in self.model_fields
        ):
            return super().__getattr__(__name)  # type: ignore
        else:
            try:
                return self._data[__name]
            except KeyError:
                raise AttributeError(
                    f"'{self.__class__.__name__}' object has no attribute '{__name}'"
                )

    def __setattr__(self, name: str, value: Any) -> None:
        if name in self.__private_attributes__ or name in self.model_fields:
            super().__setattr__(name, value)
        else:
            self._extra_train_kwargs.__setitem__(name, value)

    def __getitem__(self, key: str) -> Any:
        return self._extra_train_kwargs[key]

    def __setitem__(self, key: str, value: Any) -> None:
        self._extra_train_kwargs[key] = value

    def get(self, key: str, default: Any = None) -> Any:
        return self._extra_train_kwargs.get(key, default)

    def __contains__(self, key: str) -> bool:
        return key in self._extra_train_kwargs
