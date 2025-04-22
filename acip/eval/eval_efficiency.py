import os
import tempfile
import time
from logging import getLogger
from typing import Any

import numpy as np
import torch
from fvcore.nn import FlopCountAnalysis
from tqdm import tqdm
from transformers import PreTrainedModel, PreTrainedTokenizer, PreTrainedTokenizerFast

from acip.eval.evaluator import ModelEvaluator
from acip.eval.utils import eval_mode
from acip.model.tokenizer_factory import TokenizerFactory
from acip.utils.utils import clear_cache

logger = getLogger(__name__)


class EfficiencyModelEvaluator(ModelEvaluator):
    """
    Evaluates the efficiency of a model regarding size and inference.
    The returned dictionary has the following keys:
     - eff/model_size: The (checkpoint) size of the model in GB.
     - eff/throughput: The throughput of the model in tokens per second.
     - eff/latency: The latency of the model in seconds per batch request.
     - eff/peak_memory_allocated: The peak memory allocated by the model in GB.
     - eff/peak_memory_reserved: The peak memory reserved by the model in GB.
     - eff/flops: The FLOPs measured for the model in GigaFLOPs.
    """

    def __init__(
        self,
        tokenizer_factory: TokenizerFactory,
        eval_model_size: bool = True,
        eval_inference: bool = True,
        inference_batch_size: int = 64,
        inference_sequence_length: int = 64,
        flops_batch_size: int = 1,
        flops_sequence_length: int = 512,
    ):
        """
        Args:
            tokenizer_factory: Required to process samples in inference.
            eval_model_size: Whether to measure model size or not.
            eval_inference: Whether to evaluate model inference or not.
            inference_batch_size: See `evaluate_inference`.
            inference_sequence_length: See `evaluate_inference`.
            flops_batch_size: See `evaluate_inference`.
            flops_sequence_length: See `evaluate_inference`.
        """
        super().__init__()
        self.tokenizer = tokenizer_factory()
        self.eval_model_size = eval_model_size
        self.eval_inference = eval_inference
        self.inference_batch_size = inference_batch_size
        self.inference_sequence_length = inference_sequence_length
        self.flops_batch_size = flops_batch_size
        self.flops_sequence_length = flops_sequence_length

    def __call__(
        self,
        model: PreTrainedModel,
        prefix: str | None = "eff",
        **kwargs: Any,
    ) -> dict[str, Any]:
        prefix = prefix + "/" if prefix is not None else ""
        results = {}
        with torch.no_grad(), eval_mode(model):
            if self.eval_model_size:
                model_size_result = evaluate_model_size(model=model)
                logger.info(f"Validated Model Size:\n{model_size_result}")
                for key, result in model_size_result.items():
                    results[prefix + key] = result

            if self.eval_inference:
                inference_result = evaluate_inference(
                    model=model,
                    tokenizer=self.tokenizer,
                    inference_batch_size=self.inference_batch_size,
                    inference_sequence_length=self.inference_sequence_length,
                    flops_batch_size=self.flops_batch_size,
                    flops_sequence_length=self.flops_sequence_length,
                )
                logger.info(f"Validated Model Inference:\n{inference_result}")
                for key, result in inference_result.items():
                    results[prefix + key] = result
        return results


def evaluate_model_size(model: PreTrainedModel) -> dict[str, Any]:
    """
    Script to evaluate the memory consumption of a model.
    The model size is obtained by measuring the size of a temporary checkpoint.

    Args:
        model: A `PreTrainedModel`.

    Returns: A dictionary with key "model_size" containing the model size in GB.
    """

    logger.debug("Calculating model size...")
    with tempfile.NamedTemporaryFile(delete=True) as tmp_file:
        torch.save(model.state_dict(), tmp_file.name)
        model_size = os.path.getsize(tmp_file.name)
    return {"model_size": model_size / (1024**3)}


def evaluate_inference(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast,
    inference_batch_size: int = 64,
    inference_sequence_length: int = 64,
    inference_num_warmup_steps: int = 5,
    inference_num_steps: int = 10,
    inference_prompt: str = "Artificial intelligence is a field of computer science that",
    flops_batch_size: int = 1,
    flops_sequence_length: int = 512,
) -> dict[str, Any]:
    """
    Script to evaluate the inference latency, throughput, memory peak consumption, and FLOPs of a model.

    Args:
        model: A `PreTrainedModel` that supports causal language modeling.
        tokenizer: Suitable tokenizer for the model.
        inference_batch_size: Batch size to use for inference.
        inference_sequence_length: Length of generated text.
        inference_num_warmup_steps: Number of warm-up steps.
        inference_num_steps: Number of actual inference steps.
        inference_prompt: Prompt to be completed while running inference.
        flops_batch_size: Batch size to use for FLOPs calculation.
        flops_sequence_length: Sequence length to use for FLOPs calculation.

    Returns: Result dictionary with inference metrics.
    """

    result: dict[str, Any] = {}

    # Encode and batchify prompt.
    result["prompt"] = inference_prompt
    input_encoding = tokenizer(inference_prompt, return_tensors="pt", truncation=False)
    input_ids_single = input_encoding.input_ids
    attention_mask_single = input_encoding.attention_mask
    prompt_token_len = input_ids_single.shape[1]

    input_ids_batch = input_ids_single.repeat(inference_batch_size, 1).to(model.device)
    attention_mask_batch = attention_mask_single.repeat(inference_batch_size, 1).to(model.device)

    # Run warm-up steps.
    if inference_num_warmup_steps > 0:
        logger.debug(f"Running warm-up for {inference_num_warmup_steps} steps...")
        clear_cache(gc=True)
        for _ in tqdm(range(inference_num_warmup_steps)):
            _ = model.generate(
                input_ids=input_ids_batch,
                attention_mask=attention_mask_batch,
                max_new_tokens=inference_sequence_length,
                pad_token_id=tokenizer.pad_token_id,
                do_sample=False,
                use_cache=True,
            )
            if torch.cuda.is_available():
                torch.cuda.synchronize(model.device)

    # Reset memory stats.
    if torch.cuda.is_available():
        result["peak_memory_allocated"] = 0
        result["peak_memory_reserved"] = 0
        clear_cache(gc=True)
        torch.cuda.reset_peak_memory_stats(model.device)
    else:
        result["peak_memory_allocated"] = None
        result["peak_memory_reserved"] = None

    # Run inference and measure latency.
    result["latency"] = []
    evaluation_start_time = time.time()
    total_generated_tokens = 0

    logger.debug(f"Running inference evaluation for {inference_num_steps} steps...")
    for _ in tqdm(range(inference_num_steps)):
        start_time = time.time()
        outputs = model.generate(
            input_ids=input_ids_batch,
            attention_mask=attention_mask_batch,
            max_new_tokens=inference_sequence_length,
            pad_token_id=tokenizer.pad_token_id,
            do_sample=False,
            use_cache=True,
        )
        if torch.cuda.is_available():
            torch.cuda.synchronize(model.device)

        result["latency"].append(time.time() - start_time)
        # Calculate actual generated tokens for this batch
        num_new_tokens_per_sample = max(0, outputs.shape[1] - prompt_token_len)
        total_generated_tokens += inference_batch_size * num_new_tokens_per_sample

        del outputs

    total_evaluation_time = time.time() - evaluation_start_time

    if torch.cuda.is_available():
        result["peak_memory_allocated"] = torch.cuda.max_memory_allocated(model.device) / (1024**3)
        result["peak_memory_reserved"] = torch.cuda.max_memory_reserved(model.device) / (1024**3)

    result["latency"] = np.mean(result["latency"]).item()
    result["throughput"] = total_generated_tokens / total_evaluation_time

    # Measure FLOPs
    logger.debug("Measuring FLOPs...")
    flops_input_ids = torch.randint(
        0, tokenizer.vocab_size, (flops_batch_size, flops_sequence_length), dtype=torch.long
    ).to(model.device)
    flops_attention_mask = torch.ones(flops_batch_size, flops_sequence_length, dtype=torch.long).to(model.device)
    result["flops"] = FlopCountAnalysis(model, (flops_input_ids, flops_attention_mask)).total() / (1024**3)

    return result
