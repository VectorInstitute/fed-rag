"""Decorators"""

from .tester import TesterDecorators
from .trainer import TrainerDecorators


class Federate:
    def __init__(self, trainer: TrainerDecorators, tester: TesterDecorators):
        self.trainer = trainer
        self.tester = tester


federate = Federate(trainer=TrainerDecorators(), tester=TesterDecorators())


__all__ = ["federate"]
