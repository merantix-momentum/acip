from logging import getLogger
from typing import Any

import torch
from transformers import PreTrainedModel, PreTrainedTokenizer, PreTrainedTokenizerFast

from acip.eval.evaluator import ModelEvaluator
from acip.eval.utils import eval_mode
from acip.model.tokenizer_factory import TokenizerFactory

logger = getLogger(__name__)


class GenerationModelEvaluator(ModelEvaluator):
    """
    Evaluates a model's ability to generate text.
    The returned dictionary has the following keys:
     - gen/<idx>: the generated text for the idx-th prompt
    """

    def __init__(
        self,
        tokenizer_factory: TokenizerFactory,
        generation_prompts: list[str] | None = None,
    ):
        """
        Args:
            tokenizer_factory: Required to process samples to generate text.
            generation_prompts: A list of prompts to generate text from.
        """
        super().__init__()
        self.tokenizer = tokenizer_factory()
        self.generation_prompts = generation_prompts

    def __call__(
        self,
        model: PreTrainedModel,
        prefix: str | None = "gen",
        **kwargs: Any,
    ) -> dict[str, Any]:
        prefix = prefix + "/" if prefix is not None else ""
        results = {}
        with torch.no_grad(), eval_mode(model):
            logger.info("Validating Text Generation...")
            generation_result = evaluate_generation(
                prompts=self.generation_prompts,
                model=model,
                tokenizer=self.tokenizer,
            )
            logger.info(f"Validated Text Generation:\n{generation_result}")
            for idx, (prompt, result) in enumerate(generation_result.items()):
                results[prefix + str(idx)] = result["generated_text"]
        return results


def evaluate_generation(
    prompts: list[str],
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast,
    max_length: int = 150,
    temperature: float = 1.0,
    top_p: float = 1.0,
) -> dict[str, dict[str, Any]]:
    """
    Script to evaluate text generation.

    Args:
        prompts: List of prompts to generate text from.
        model: A `PreTrainedModel` that supports causal language modeling.
        tokenizer: Suitable tokenizer for the model.
        max_length: Maximum length of the text to be generated.
        temperature: Sampling temperature.
        top_p: Top-p sampling.

    Returns: Nested dictionary of generated text in the format [<prompt>]["generated_text"/"num_tokens"].
    """
    result = {}
    for prompt in prompts:
        logger.debug(f"Completing prompt: '{prompt}'")
        encodings = tokenizer(prompt, return_tensors="pt")

        output = model.generate(
            encodings.input_ids.to(model.device),
            max_new_tokens=max_length,
            num_return_sequences=1,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )
        generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
        result[prompt] = {
            "generated_text": generated_text,  # contains prompt as well
            "num_tokens": len(output[0]),
        }
    return result
