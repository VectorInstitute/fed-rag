"""HuggingFace PeftModel Generator"""

from typing import TYPE_CHECKING, Any, Optional

import torch
import torch.nn.functional as F
from pydantic import ConfigDict, Field, PrivateAttr

try:
    from peft import PeftModel, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM
    from transformers.generation.utils import GenerationConfig

    _has_huggingface = True
except ModuleNotFoundError:
    _has_huggingface = False


if TYPE_CHECKING:  # pragma: no cover
    from peft import PeftModel
    from transformers.generation.utils import GenerationConfig

from fed_rag.base.generator import BaseGenerator
from fed_rag.tokenizers.hf_pretrained_tokenizer import HFPretrainedTokenizer

DEFAULT_PROMPT_TEMPLATE = """
You are a helpful assistant. Given the user's question, provide a succinct
and accurate response. If context is provided, use it in your answer if it helps
you to create the most accurate response.

<question>
{question}
</question>

<context>
{context}
</context>

<response>

"""


class HFPeftModelGenerator(BaseGenerator):
    """HFPeftModelGenerator Class.

    NOTE: this class supports loading PeftModel's from HF Hub or from local.
    TODO: support loading custom models via a `~peft.Config` and `~peft.get_peft_model`
    """

    model_config = ConfigDict(protected_namespaces=("pydantic_model_",))
    model_name: str = Field(
        description="Name of Peft model. Used for loading model from HF hub or local."
    )
    base_model_name: str = Field(
        description="Name of the frozen HuggingFace base model. Used for loading the model from HF hub or local."
    )
    generation_config: "GenerationConfig" = Field(
        description="The generation config used for generating with the PreTrainedModel."
    )
    load_model_kwargs: dict = Field(
        description="Optional kwargs dict for loading peft model from HF. Defaults to None.",
        default_factory=dict,
    )
    load_base_model_kwargs: dict = Field(
        description="Optional kwargs dict for loading base model from HF. Defaults to None.",
        default_factory=dict,
    )
    prompt_template: str = Field(description="Prompt template for RAG.")
    _model: Optional["PeftModel"] = PrivateAttr(default=None)
    _tokenizer: HFPretrainedTokenizer | None = PrivateAttr(default=None)

    def __init__(
        self,
        model_name: str,
        base_model_name: str,
        generation_config: Optional["GenerationConfig"] = None,
        prompt_template: str | None = None,
        load_model_kwargs: dict | None = None,
        load_base_model_kwargs: dict | None = None,
        load_model_at_init: bool = True,
    ):
        if not _has_huggingface:
            msg = (
                f"`{self.__class__.__name__}` requires `huggingface` extra to be installed. "
                "To fix please run `pip install fed-rag[huggingface]`."
            )
            raise ValueError(msg)

        generation_config = (
            generation_config if generation_config else GenerationConfig()
        )
        prompt_template = (
            prompt_template if prompt_template else DEFAULT_PROMPT_TEMPLATE
        )
        super().__init__(
            model_name=model_name,
            base_model_name=base_model_name,
            generation_config=generation_config,
            prompt_template=prompt_template,
            load_model_kwargs=load_model_kwargs if load_model_kwargs else {},
            load_base_model_kwargs=(
                load_base_model_kwargs if load_base_model_kwargs else {}
            ),
        )
        self._tokenizer = HFPretrainedTokenizer(
            model_name=base_model_name, load_model_at_init=load_model_at_init
        )
        if load_model_at_init:
            self._model = self._load_model_from_hf()

    def _load_model_from_hf(self, **kwargs: Any) -> "PeftModel":
        load_base_kwargs = self.load_base_model_kwargs
        load_kwargs = self.load_model_kwargs
        load_kwargs.update(kwargs)
        self.load_model_kwargs = load_kwargs  # update load_model_kwargs
        base_model = AutoModelForCausalLM.from_pretrained(
            self.base_model_name, **load_base_kwargs
        )

        if "quantization_config" in load_base_kwargs:
            # preprocess model for kbit fine-tuning
            # https://huggingface.co/docs/peft/developer_guides/quantization
            base_model = prepare_model_for_kbit_training(base_model)

        return PeftModel.from_pretrained(
            base_model, self.model_name, **load_kwargs
        )

    @property
    def model(self) -> "PeftModel":
        if self._model is None:
            # load HF PeftModel
            self._model = self._load_model_from_hf()
        return self._model

    @model.setter
    def model(self, value: "PeftModel") -> None:
        self._model = value

    @property
    def tokenizer(self) -> HFPretrainedTokenizer:
        return self._tokenizer

    @tokenizer.setter
    def tokenizer(self, value: HFPretrainedTokenizer) -> None:
        self._tokenizer = value

    # generate
    def generate(self, query: str, context: str, **kwargs: Any) -> str:
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
        self, prompt: str, target: str
    ) -> torch.Tensor:
        """Computes the target sequence probability given the prompt.

        Args:
            generator (BaseGenerator): The generator LLM
            prompt (str): The input i.e. conditional prompt sequence
            target (str): The target sequence

        Returns:
            proba (torch.Tensor): The probability of target sequence given a prompt.
                i.e., P_{LLM}(target | sequence)
        """
        # Combine prompt and target for teacher forcing
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
            next_token_pos = target_start_idx + i - 1
            next_token_logits = logits[0, next_token_pos, :]
            probs = F.softmax(next_token_logits, dim=-1)
            log_prob = torch.log(probs[target_id]).item()
            log_probs.append(log_prob)

        # Sum log probabilities to get sequence log probability
        sequence_log_prob = sum(log_probs)
        # Convert to probability
        sequence_prob = torch.exp(torch.tensor(sequence_log_prob))

        return sequence_prob
