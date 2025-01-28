"""Decorators"""

from .tester import TesterDecorators
from .trainer import TrainerDecorators

trainer = TrainerDecorators()
tester = TesterDecorators()

__all__ = ["trainer", "tester"]
