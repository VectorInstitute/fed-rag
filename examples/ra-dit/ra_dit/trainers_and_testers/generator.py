"""Generator trainer and tester following RA-DIT."""

# torch
import torch

# hf
from datasets import Dataset, load_dataset
from torch.types import Device
from transformers import PreTrainedModel
from transformers.trainer_utils import TrainOutput
from trl import SFTConfig, SFTTrainer

# fedrag
from fed_rag.decorators import federate
from fed_rag.types import TestResult, TrainResult

# Dataset
train_dataset = load_dataset("stanfordnlp/imdb", split="train[:20]")
val_dataset = load_dataset("stanfordnlp/imdb", split="test[:10]")


@federate.trainer.huggingface
def generator_train_loop(
    model: PreTrainedModel,
    train_data: Dataset,
    val_data: Dataset,
    device: Device,
) -> TrainResult:
    """RA-DIT training loop for generator."""

    model.to(device)
    training_args = SFTConfig(
        max_seq_length=512,
        output_dir="~/scratch/tmp",
    )
    trainer = SFTTrainer(
        model,
        train_dataset=train_data,
        args=training_args,
        eval_dataset=val_data,
    )

    train_output: TrainOutput = trainer.train()

    return TrainResult(loss=train_output.training_loss)


@federate.tester.huggingface
def generator_evaluate(m: PreTrainedModel, test_data: Dataset) -> TestResult:
    return TestResult(loss=42.0, metrics={})


if __name__ == "__main__":
    from ra_dit.generators.llama2_7b import generator

    train_result = generator_train_loop(
        model=generator.model,
        train_data=train_dataset,
        val_data=val_dataset,
        device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
    )
    print(train_result)
