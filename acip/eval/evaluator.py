from logging import getLogger
from typing import Any

from transformers import PreTrainedModel

from acip.utils.utils import clear_cache

logger = getLogger(__name__)


class ModelEvaluator:
    """
    Base class for a model evaluator. Call an instance to perform the evaluation.
    This base class also serves as a dummy evaluator that returns an empty results dictionary.
    """

    def __call__(
        self,
        model: PreTrainedModel,
        prefix: str | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """
        Performs the evaluation of a `PreTrainedModel`.

        Args:
            model: The model to evaluate.
            prefix: If not None, each key in the results dictionary will be prefixed, where "/" is
                use as separator to the actual key.
            **kwargs: Optional keyword arguments than can be used by the evaluator.

        Returns: A dictionary of results. Each key represents a metric that is optionally prefixed by `prefix`.
        """
        return {}


class ComposedModelEvaluator(ModelEvaluator):
    """
    A model evaluator that wraps multiple `ModelEvaluator` instances and executes them sequentially.
    """

    def __init__(self, evaluators: dict[str, ModelEvaluator]):
        """
        Args:
            evaluators: Dictionary of `ModelEvaluator` instances to be executed sequentially.
        """
        self.evaluators = evaluators

    def __call__(
        self,
        model: PreTrainedModel,
        prefix: str | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """
        Calls each evaluator in `self.evaluators` sequentially.

        Args:
            model: The model to evaluate.
            prefix: If not None, each key in the results dictionary will be prefixed, where "/" is
                use as separator to the actual key. If the wrapped evaluators have a prefix on their own,
                it will be kept, leading to a hierarchy of prefixes separated by "/".
            **kwargs: Optional keyword arguments than can be used by the evaluator.

        Returns: A dictionary of results. Each key represents a metric that is optionally prefixed by `prefix`.
        """
        prefix = prefix + "/" if prefix is not None else ""
        results = {}
        for idx, evaluator in enumerate(self.evaluators.values()):
            logger.info(f"Running evaluator {idx}/{len(self.evaluators)}")
            evaluator_result = evaluator(model)
            for k, result in evaluator_result.items():
                results[prefix + k] = result

        clear_cache()
        return results
