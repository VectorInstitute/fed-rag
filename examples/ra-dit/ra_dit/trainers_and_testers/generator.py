"""Generator trainer and tester following RA-DIT."""

from datasets import Dataset, load_dataset
from peft import PeftConfig
from torch.types import Device
from transformers import PreTrainedModel
from transformers.trainer_utils import TrainOutput
from trl import SFTConfig, SFTTrainer

from fed_rag.decorators import federate
from fed_rag.types import TestResult, TrainResult
from fed_rag.types.rag_system import RAGSystem

# Dataset
train_dataset = load_dataset("stanfordnlp/imdb", split="train[:20]")
val_dataset = load_dataset("stanfordnlp/imdb", split="test[:10]")


# Custom SFTTrainer to implement RA-IT
class RetrievalAugmentedSFTTrainer(SFTTrainer):
    """A custom SFTTrainer to implement retrieval-augmented instruction fine-tuning."""

    def compute_loss(self, model: PreTrainedModel, inputs, return_outputs=False):
        try:
            rag_system: RAGSystem = model.__associated_rag_system
        except AttributeError:
            raise ValueError(
                "Unable to get associated RAGSystem with supplied `BaseGenerator`."
            )
        questions = inputs.pop("questions")
        answers = inputs.pop("answers")

        # retrieve
        for q in questions:
            source_nodes = rag_system.retrieve(query=q)
            # parallel in-context retrieval augmentation
            proba_tensors = []
            for n in source_nodes:
                context = n.node.text_content
                # prepare prompt
                prompt = rag_system.generator.prompt_template.format(
                    question=q, context=context
                )
                # probas
                outputs = model(prompt)
                logits = outputs.logits
                probas = ...
                proba_tensors.append(probas)
            weighted_probas = ...

            # loss
            loss = ...  # masked language model

        return (loss, outputs) if return_outputs else loss


@federate.trainer.huggingface
def generator_train_loop(
    model: PreTrainedModel,
    train_data: Dataset,
    val_data: Dataset,
    device: Device | None = None,
    peft_config: PeftConfig | None = None,
) -> TrainResult:
    """RA-DIT training loop for generator."""

    if device:
        model = model.to(device)

    training_args = SFTConfig(
        max_seq_length=512,
        output_dir="~/scratch/tmp",
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
    return TestResult(loss=42.0, metrics={})


if __name__ == "__main__":
    from ra_dit.generators.llama2_7b import generators

    # by default we load generator to cpu since fine-tuning just one generator
    # we instead set `device_map` to `auto`
    generator = generators["qlora"]

    train_result = generator_train_loop(
        model=generator.model,
        train_data=train_dataset,
        val_data=val_dataset,
        peft_config=generator.model.active_peft_config,
    )
    print(train_result)
