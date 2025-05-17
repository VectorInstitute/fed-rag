"""Base Generator"""

from abc import ABC, abstractmethod

import torch
from pydantic import BaseModel, ConfigDict

from fed_rag.base.tokenizer import BaseTokenizer

DEFAULT_PROMPT_TEMPLATE = """
You are a helpful assistant. Given the user's query, provide a succinct
and accurate response. If context is provided, use it in your answer if it helps
you to create the most accurate response.

<query>
{query}
</query>

<context>
{context}
</context>

<response>

"""


class BaseGenerator(BaseModel, ABC):
    """Base Generator Class."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @abstractmethod
    def generate(self, query: str, context: str, **kwargs: dict) -> str:
        """Generate an output from a given query and context."""

    @property
    @abstractmethod
    def model(self) -> torch.nn.Module:
        """Model associated with this generator."""

    @property
    @abstractmethod
    def tokenizer(self) -> BaseTokenizer:
        """Tokenizer associated with this generator."""

    @abstractmethod
    def compute_target_sequence_proba(
        self, prompt: str, target: str
    ) -> torch.Tensor:
        """Compute P(target | prompt).

        NOTE: this is used in LM Supervised Retriever fine-tuning.
        """

    @property
    @abstractmethod
    def prompt_template(self) -> str:
        """Prompt template for formating query and context."""

    @prompt_template.setter
    @abstractmethod
    def prompt_template(self, value: str) -> None:
        """Prompt template setter."""
