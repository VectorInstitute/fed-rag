"""Generator trainer and tester following RA-DIT."""

from datasets import Dataset, load_dataset
from peft import PeftConfig
from torch.types import Device
from transformers import PreTrainedModel
from transformers.trainer_utils import TrainOutput
from trl import SFTConfig, SFTTrainer

from fed_rag.decorators import federate
from fed_rag.types import TestResult, TrainResult

from ..utils import generate_timestamp

# Dataset
train_dataset = load_dataset("stanfordnlp/imdb", split="train[:20]")
val_dataset = load_dataset("stanfordnlp/imdb", split="test[:10]")


@federate.trainer.huggingface
def generator_train_loop(
    model: PreTrainedModel,
    train_data: Dataset,
    val_data: Dataset,
    device: Device | None = None,
    peft_config: PeftConfig | None = None,
    checkpoint_dir: str | None = ".checkpoints/generator",
) -> TrainResult:
    """RA-DIT training loop for generator."""

    if device:
        model = model.to(device)

    training_args = SFTConfig(
        max_seq_length=512,
        output_dir=checkpoint_dir,
        run_name=generate_timestamp(),
    )
    trainer = SFTTrainer(
        model,
        train_dataset=train_data,
        args=training_args,
        eval_dataset=val_data,
        peft_config=peft_config,
    )

    train_output: TrainOutput = trainer.train()

    return TrainResult(loss=train_output.training_loss)


@federate.tester.huggingface
def generator_evaluate(m: PreTrainedModel, test_data: Dataset) -> TestResult:
    training_args = SFTConfig(
        max_seq_length=512,
        do_train=False,
    )
    trainer = SFTTrainer(m, args=training_args, train_dataset=test_data)
    eval_results = trainer.evaluate(trainer.train_dataset)
    return TestResult(loss=eval_results.get("eval_loss"), metrics={})


if __name__ == "__main__":
    from ra_dit.generators.llama2_7b import generator_registry

    # by default we load generator to cpu since fine-tuning just one generator
    # we instead set `device_map` to `auto`
    generator = generator_registry["qlora"]

    train_result = generator_train_loop(
        model=generator.model,
        train_data=train_dataset,
        val_data=val_dataset,
        peft_config=generator.model.active_peft_config,
    )
    test_result = generator_evaluate(generator.model, val_dataset)

    print(test_result)
    print(test_result)
