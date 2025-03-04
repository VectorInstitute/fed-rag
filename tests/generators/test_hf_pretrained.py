from fed_rag.base.generator import BaseGenerator
from fed_rag.generators.hf_pretrained_model import HFPretrainedModelGenerator


def test_hf_pretrained_generator_class() -> None:
    names_of_base_classes = [
        b.__name__ for b in HFPretrainedModelGenerator.__mro__
    ]
    assert BaseGenerator.__name__ in names_of_base_classes
