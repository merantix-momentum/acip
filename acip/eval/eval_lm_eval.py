from logging import getLogger
from typing import Any

import lm_eval.models.huggingface
import torch
from lm_eval import evaluator
from transformers import PreTrainedModel, PreTrainedTokenizer, PreTrainedTokenizerFast

from acip.eval.evaluator import ModelEvaluator
from acip.eval.utils import eval_mode
from acip.model.tokenizer_factory import TokenizerFactory

logger = getLogger(__name__)


class LMEvalModelEvaluator(ModelEvaluator):
    """
    Evaluates a model on serval zero/few-shot tasks using EleutherAI's Language Model Evaluation Harness (lm_eval).
    The returned dictionary has the following keys:
     - lm_eval/<task_name>: The average accuracy of the model on <task_name>.
     - lm_eval/average: The average accuracy of the model over all tasks.
    """

    def __init__(
        self,
        tokenizer_factory: TokenizerFactory,
        task_names: tuple[str] | list[str],
        batch_size: int | None = 64,
        num_fewshots: int = 0,
    ):
        """
        Args:
            tokenizer_factory: Required to process samples of the datasets.
            task_names: Names of tasks to evaluate on.
                See https://github.com/EleutherAI/lm-evaluation-harness/tree/main/lm_eval/tasks for a list
                of available tasks.
            batch_size: Batch size to use for evaluation.
            num_fewshots: Number of few-shot examples to use. Defaults to 0, i.e., zero-shot.
        """
        super().__init__()
        self.tokenizer = tokenizer_factory()
        self.task_names = task_names
        self.batch_size = batch_size
        self.num_fewshots = num_fewshots

    def __call__(
        self,
        model: PreTrainedModel,
        prefix: str | None = "lm_eval",
        **kwargs: Any,
    ) -> dict[str, Any]:
        prefix = prefix + "/" if prefix is not None else ""
        results = {}
        with torch.no_grad(), eval_mode(model):
            logger.info(f"Validating LM Eval on {self.task_names}...")
            lm_eval_result = evaluate_lm_eval(
                task_names=self.task_names,
                model=model,
                tokenizer=self.tokenizer,
                batch_size=self.batch_size,
                num_fewshot=self.num_fewshots,
            )
            logger.info(f"Validated LM Eval:\n{lm_eval_result}")
            for task_name, result in lm_eval_result.items():
                results[prefix + task_name] = result["acc"]
        return results


def evaluate_lm_eval(
    task_names: list[str],
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast,
    num_fewshot: int = 0,
    batch_size: int | None = 64,
) -> dict[str, dict[str, float]]:
    """
    Script to evaluate a model on a list of tasks using lm_eval.

    Args:
        task_names: Names of tasks to evaluate on.
        model: An `PreTrainedModel` that supports causal language modeling.
        tokenizer: Suitable tokenizer for the model.
        num_fewshot: Number of few-shot examples to use. Defaults to 0, i.e., zero-shot.
        batch_size: Batch size to use for evaluation.

    Returns: Nested dictionary of results in the format [<task_name>]["acc"/"acc_stderr"], reporting accuracy and
        accuracy standard error for each task, respectively.
    """
    wrapped_model = lm_eval.models.huggingface.HFLM(
        pretrained=model,
        tokenizer=tokenizer,
        device=None,  # device of the model is used
        trust_remote_code=True,
        batch_size=batch_size,
    )
    lm_eval_result = evaluator.simple_evaluate(
        model=wrapped_model,
        tasks=task_names,
        num_fewshot=num_fewshot,
        batch_size=None,
        use_cache=None,
        check_integrity=False,
    )
    results = {}
    for task_name, result in lm_eval_result["results"].items():
        results[task_name] = {
            "acc": result["acc,none"],
            "acc_stderr": result["acc_stderr,none"],
        }
    results["average"] = {
        "acc": sum([results[task_name]["acc"] for task_name in results]) / len(results),
    }

    return results
