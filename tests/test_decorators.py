"""Decorators unit tests"""

from typing import Any

import torch.nn as nn

from fed_rag.decorators import federate


def test_decorated_trainer() -> None:
    def fn(net: nn.Module) -> Any:
        pass

    decorated = federate.trainer.pytorch(fn)
    config = getattr(decorated, "__fl_task_trainer_config")
    assert len(config) == 0


def test_decorated_tester() -> None:
    def fn(net: nn.Module) -> Any:
        pass

    decorated = federate.tester.pytorch(fn)
    config = getattr(decorated, "__fl_task_tester_config")
    assert len(config) == 0
