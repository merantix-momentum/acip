from logging import getLogger
from typing import Any, Type

from transformers import PretrainedConfig, PreTrainedModel

from acip.core.acip_model import ACIPModel, ACIPPruningConfig
from acip.core.utils import get_class_from_str

logger = getLogger(__name__)


class PretrainedModelFactory:
    """
    A factory for creating Huggingface `PreTrainedModel` instances.
    The concept of lazy creation is useful in `BaseLitModule`, where the model created with the `configure_model` hook.
    This allows for more advanced training strategies, such as model parallelism.
    """

    def __init__(
        self,
        pretrained_model_cls: Type[PreTrainedModel],
        pretrained_model_name_or_path: str | None = None,
        pretrained_model_config: PretrainedConfig | None = None,
        pretrained_model_kwargs: dict[str, Any] | None = None,
    ):
        """
        Args:
            pretrained_model_cls: Class of the pretrained model,
            pretrained_model_name_or_path: Name or path of the pretrained model used in `from_pretrained`.
            pretrained_model_config: Optional config passed to `from_pretrained` or the model constructor.
            pretrained_model_kwargs: Keyword arguments passed to `from_pretrained` or the model constructor.
        """
        if isinstance(pretrained_model_cls, str):
            # Resolve class from string if applicable
            self.pretrained_model_cls = get_class_from_str(pretrained_model_cls)  # noqa
        else:
            self.pretrained_model_cls = pretrained_model_cls
        self.pretrained_model_name_or_path = pretrained_model_name_or_path
        self.pretrained_model_config = pretrained_model_config
        self.pretrained_model_kwargs = pretrained_model_kwargs if pretrained_model_kwargs is not None else {}

    def __call__(self, **kwargs: Any) -> PreTrainedModel:
        """Create the model. See `self.create_model` for details."""
        return self.create_model()

    def create_model(self, **kwargs: Any) -> PreTrainedModel:
        """
        Create the model. If `pretrained_model_name_or_path` is provided, the model is loaded `from_pretrained`,
        otherwise the model is created from scratch.

        Args:
            kwargs: Additional keyword arguments passed to `from_pretrained` or the model constructor.
                Will be merged with `pretrained_model_kwargs`.

        Returns: The created model.
        """
        pretrained_model_kwargs = self.pretrained_model_kwargs.copy()
        pretrained_model_kwargs.update(kwargs)
        if self.pretrained_model_name_or_path is not None:
            logger.debug(f"Loading pretrained model from {self.pretrained_model_name_or_path}.")
            return self.pretrained_model_cls.from_pretrained(
                self.pretrained_model_name_or_path, config=self.pretrained_model_config, **pretrained_model_kwargs
            )
        else:
            logger.debug("Creating pretrained model from scratch.")
            return self.pretrained_model_cls(self.pretrained_model_config, **pretrained_model_kwargs)


class ACIPModelFactory(PretrainedModelFactory):
    """
    A factory for creating ACIP models. Extending `PretrainedModelFactory`, it provides options to post-process
    the created model before returning it, namely, pruning & compression or quantization.
    Note that some of these operations cannot be reverted.
    """

    def __init__(
        self,
        pretrained_model_cls: Type[PreTrainedModel],
        pretrained_model_name_or_path: str | None = None,
        pretrained_model_config: PretrainedConfig | None = None,
        pretrained_model_kwargs: dict[str, Any] | None = None,
        prune_to_ratio: float | None = None,
        compress_and_unparametrize: bool = True,
        measure_ratio_full: bool = False,
        pruning_config: ACIPPruningConfig | None = None,
        quantize_weights: bool = False,
    ):
        """
        Args:
            pretrained_model_cls: Class of the pretrained model,
            pretrained_model_name_or_path: Name or path of the pretrained model used in `from_pretrained`.
            pretrained_model_config: Optional config passed to `from_pretrained` or the model constructor.
            pretrained_model_kwargs: Keyword arguments passed to `from_pretrained` or the model constructor.
            prune_to_ratio: If a float between 0 and 1 is provided, the created model will be directly pruned
                according to this size ratio. See `ACIPModel.prune_model_by_score`.
            compress_and_unparametrize: If True, the parametrized modules of the created model will compressed
                according to their pruning state (controlled by `prune_to_ratio`) and non-compressible modules
                will be unparametrized. See `ParametrizedModel.compress`. Note that compared to `prune_to_ratio`,
                this operation cannot be reverted.
            measure_ratio_full: If `True`, all parameters of the model are counted when pruning to a target ratio
                is performed, if `False` only the parameters of the parametrized modules are counted (default).
                See `full` flag in `ACIPModel.prune_model_by_score`.
            pruning_config: Optional config for the pruning process passed to `ACIPModel.prune_model_by_score`.
            quantize_weights: If True, the created model will be directly quantized. See `ParametrizedModel.quantize`.
                Note that this operation cannot be reverted.
        """
        super().__init__(
            pretrained_model_cls, pretrained_model_name_or_path, pretrained_model_config, pretrained_model_kwargs
        )
        self.prune_to_ratio = prune_to_ratio
        self.compress_and_unparametrize = compress_and_unparametrize
        self.measure_ratio_full = measure_ratio_full
        self.pruning_config = pruning_config
        self.quantize_weights = quantize_weights

    def create_model(self) -> ACIPModel:
        model = super().create_model()
        assert isinstance(model, ACIPModel), "Model must be of type ACIPModel"

        if self.prune_to_ratio is not None:
            # Prune target params to desired size ratio ...
            model.prune_model_by_score(
                size_ratio=self.prune_to_ratio, full=self.measure_ratio_full, pruning_config=self.pruning_config
            )
            logger.info(f"Pruned model to size ratio {self.prune_to_ratio}.")

            # ... and compress.
            if self.compress_and_unparametrize:
                model.compress()
                logger.info("Model compressed and unparametrized where possible.")

        if self.quantize_weights:
            model.quantize()
            logger.info("Model quantized.")

        return model
