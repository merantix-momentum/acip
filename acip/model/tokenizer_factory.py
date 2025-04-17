from abc import ABC, abstractmethod
from typing import Any

from transformers import AutoTokenizer, LlamaTokenizer, PreTrainedTokenizer, PreTrainedTokenizerFast


class TokenizerFactory(ABC):
    """
    A factory base class for creating Huggingface tokenizers.
    The concept of lazy creation is useful because it separates the tokenizer creation from hydra instantiations
    in the entrypoint.
    """

    @abstractmethod
    def create_tokenizer(self) -> PreTrainedTokenizer | PreTrainedTokenizerFast:
        """Creates the tokenizer."""
        raise NotImplementedError

    def __call__(self) -> PreTrainedTokenizer | PreTrainedTokenizerFast:
        """Creates the tokenizer."""
        return self.create_tokenizer()


class AutoTokenizerFactory(TokenizerFactory):
    """
    Creates Huggingface tokenizers via `AutoTokenizer`.
    """

    def __init__(self, model_name_or_path: str, tokenizer_kwargs: dict[str, Any] | None = None):
        """
        Args:
            model_name_or_path: The model name or path associated with the tokenizer.
                See `AutoTokenizer.from_pretrained`.
            tokenizer_kwargs: Keyword arguments passed to `AutoTokenizer.from_pretrained`.
        """
        self.model_name_or_path = model_name_or_path
        self.tokenizer_kwargs = tokenizer_kwargs if tokenizer_kwargs is not None else {}

    def create_tokenizer(self) -> PreTrainedTokenizer | PreTrainedTokenizerFast:
        tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path, **self.tokenizer_kwargs)
        tokenizer.pad_token = tokenizer.eos_token  # standard in causal language modeling
        return tokenizer


class LlamaTokenizerFactory(AutoTokenizerFactory):
    """
    Creates a `LlamaTokenizer` analogously to `AutoTokenizerFactory`.
    Required because older LLaMA models (1st generation) may not support `AutoTokenizer`.
    """

    def create_tokenizer(self) -> LlamaTokenizer:
        tokenizer = LlamaTokenizer.from_pretrained(self.model_name_or_path, **self.tokenizer_kwargs)
        tokenizer.pad_token = tokenizer.eos_token  # standard in causal language modeling
        return tokenizer
