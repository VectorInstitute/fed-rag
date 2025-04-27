from fed_rag.base.rag_trainer import BaseRAGTrainer
from fed_rag.rag_trainers.pytorch import PyTorchRAGTrainer


def test_pt_rag_trainer_class() -> None:
    names_of_base_classes = [b.__name__ for b in PyTorchRAGTrainer.__mro__]
    assert BaseRAGTrainer.__name__ in names_of_base_classes
