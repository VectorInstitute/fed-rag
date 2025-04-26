"""HuggingFace Generator Mixin."""

from typing import TYPE_CHECKING, Any, Protocol, Union, runtime_checkable

import torch
import torch.nn.functional as F

if TYPE_CHECKING:  # pragma: no cover
    from peft import PeftModel
    from transformers import PreTrainedModel
    from transformers.generation.utils import GenerationConfig

from fed_rag.tokenizers.hf_pretrained_tokenizer import HFPretrainedTokenizer


@runtime_checkable
class HFGeneratorProtocol(Protocol):
    prompt_template: str
    tokenizer: HFPretrainedTokenizer
    model: Union["PreTrainedModel", "PeftModel"]
    generation_config: "GenerationConfig"


class HuggingFaceGeneratorMixin:
    # generate
    def generate(
        self: HFGeneratorProtocol, query: str, context: str, **kwargs: Any
    ) -> str:
        formatted_query = self.prompt_template.format(
            question=query, context=context
        )

        # encode query
        tokenizer_result = self.tokenizer.unwrapped(
            formatted_query, return_tensors="pt"
        )
        inputs: torch.Tensor = tokenizer_result.input_ids
        inputs = inputs.to(self.model.device)

        # generate
        generated_ids = self.model.generate(
            inputs=inputs,
            generation_config=self.generation_config,
            tokenizer=self.tokenizer.unwrapped,
            **kwargs,
        )

        # decode tokens
        outputs: list[str] = self.tokenizer.unwrapped.batch_decode(
            generated_ids, skip_special_tokens=True
        )
        return outputs[0]

    def compute_target_sequence_proba(
        self: HFGeneratorProtocol, prompt: str, target: str
    ) -> torch.Tensor:
        """Computes the target sequence probability given the prompt.

        Args:
            generator (BaseGenerator): The generator LLM
            prompt (str): The input i.e. conditional prompt sequence
            target (str): The target sequence

        Returns:
            proba (torch.Tensor): The probability of target sequence given a prompt.
                i.e., P_{LLM}(target | prompt)
        """
        input_text = prompt + target
        encode_result = self.tokenizer.encode(input_text)
        input_ids = encode_result["input_ids"]

        # Get the token IDs for just the target portion
        prompt_only_encode_result = self.tokenizer.encode(prompt)
        target_start_idx = len(prompt_only_encode_result["input_ids"])
        target_ids = input_ids[target_start_idx:]

        # Get the logits from the model
        with torch.no_grad():
            outputs = self.model(torch.tensor(input_ids).unsqueeze(0))
            logits = outputs.logits

        # Calculate probability of each target token given the previous tokens
        log_probs = []
        for i, target_id in enumerate(target_ids):
            # get log prob of next target token in the sequence
            next_token_pos = target_start_idx + i
            next_token_logits = logits[0, next_token_pos, :]
            probs = F.softmax(next_token_logits, dim=-1)
            log_prob = torch.log(probs[target_id]).item()
            log_probs.append(log_prob)

        # Sum log probabilities to get sequence log probability
        sequence_log_prob = sum(log_probs)
        # Convert to probability
        sequence_prob = torch.exp(torch.tensor(sequence_log_prob))

        return sequence_prob
