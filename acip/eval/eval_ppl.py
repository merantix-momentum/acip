from logging import getLogger
from typing import Any

import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import PreTrainedModel, PreTrainedTokenizer, PreTrainedTokenizerFast

from acip.eval.evaluator import ModelEvaluator
from acip.eval.utils import eval_mode
from acip.model.tokenizer_factory import TokenizerFactory

logger = getLogger(__name__)


class PPLModelEvaluator(ModelEvaluator):
    """
    Evaluates the perplexity of a model on given test dataset.
    The returned dictionary has the following keys:
     - ppl/<dataset_name>: The average perplexity of the model on <dataset_name>
    """

    def __init__(
        self, tokenizer_factory: TokenizerFactory, dataset_names: tuple[str] | list[str], ctx_length: int | None = None
    ):
        """
        Args:
            tokenizer_factory: Required to process samples of the datasets.
            dataset_names: Names of the datasets to evaluate on. Currently supported: "wikitext", "ptb", "c4".
            ctx_length: Each dataset is fed to the model in chunks of this length.
                If None, the model's max context length is used.
        """
        super().__init__()
        self.tokenizer = tokenizer_factory()
        self.dataset_names = dataset_names
        self.ctx_length = ctx_length

    def __call__(
        self,
        model: PreTrainedModel,
        prefix: str | None = "ppl",
        **kwargs: Any,
    ) -> dict[str, Any]:
        prefix = prefix + "/" if prefix is not None else ""
        results = {}
        with torch.no_grad(), eval_mode(model):
            for dataset_name in self.dataset_names:
                logger.info(f"Validating PPL on {dataset_name}...")
                ppl = evaluate_ppl(
                    dataset_name=dataset_name, model=model, tokenizer=self.tokenizer, ctx_length=self.ctx_length
                )
                logger.info(f"Validated PPL on {dataset_name}: {ppl}")
                results[prefix + dataset_name] = ppl
        return results


def evaluate_ppl(
    dataset_name: str,
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast,
    ctx_length: int | None,
) -> float:
    """
    Script to evaluate the perplexity of a model on a given dataset where the samples are fed to the model in chunks.

    Adapted from https://github.com/locuslab/wanda/blob/main/lora_ft/evaluate_ppl.py.

    Args:
        dataset_name: Name of the dataset to evaluate on. Currently supported: "wikitext", "ptb", "c4".
        model: An `PreTrainedModel` that supports causal language modeling.
        tokenizer: Suitable tokenizer for the model.
        ctx_length: Chunk size to feed to the model. If None, the model's max context length is used.

    Returns: The average perplexity of the model on the dataset.
    """
    # Load datasets and concatenate all samples into a single string and encode it via the tokenizer
    if dataset_name == "wikitext":
        test = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
        encodings = tokenizer("\n\n".join(test["text"]), return_tensors="pt")
    elif dataset_name == "ptb":
        testdata = load_dataset("ptb_text_only", "penn_treebank", split="test")
        encodings = tokenizer(" ".join(testdata["sentence"]), return_tensors="pt")
    elif dataset_name == "c4":
        valdata = load_dataset("allenai/c4", "en", split="validation", streaming=True)
        text = []
        n_samples = 1100
        for idx, example in tqdm(enumerate(iter(valdata)), total=n_samples, desc="Loading C4 data"):
            if idx < n_samples:
                text.append(example["text"])
            else:
                break
        encodings = tokenizer(" ".join(text), return_tensors="pt")
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    # Get max context length of model if ctx_length is not provided
    if ctx_length is None:
        ctx_length = model.config.max_position_embeddings

    # Feed the encoded string to the model in chunks of size ctx_length and measure the perplexity on each chunk
    max_length = ctx_length
    stride = ctx_length
    seq_length = encodings.input_ids.size(1)
    nlls = []
    prev_end_loc = 0
    for begin_loc in tqdm(range(0, seq_length, stride)):
        end_loc = min(begin_loc + max_length, seq_length)
        trg_len = end_loc - prev_end_loc  # may be different from stride on last loop
        input_ids = encodings.input_ids[:, begin_loc:end_loc].to(next(model.parameters()).device)
        target_ids = input_ids.clone()
        target_ids[:, :-trg_len] = -100

        with torch.no_grad():
            outputs = model(input_ids, labels=target_ids)
            neg_log_likelihood = outputs.loss

        nlls.append(neg_log_likelihood)

        prev_end_loc = end_loc
        if end_loc >= seq_length:
            break

    ppl = torch.exp(torch.stack(nlls).mean())
    return ppl.item()
