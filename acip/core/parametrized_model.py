import logging
import os
from dataclasses import asdict, dataclass, field
from typing import Any, Literal, Type

import torch
from peft import PeftConfig
from peft.tuners.tuners_utils import _maybe_include_all_linear_layers, check_target_module_exists
from torch import nn
from transformers import AutoConfig, PretrainedConfig, PreTrainedModel

from .parametrized_layer import Parametrization, parametrize_module, ParametrizedModule, unparametrize_module
from .projected_layer import SVDLinearParametrization
from .utils import get_class_from_str, get_str_from_class, init_empty_weights

logger = logging.getLogger(__name__)


@dataclass
class BaseModelConfig:
    """
    Configuration for the base model to be parametrized by `ParametrizedModel`.

    Attributes:
        pretrained_model_cls: The class of the base model. Child class of `PreTrainedModel`.
        pretrained_model_kwargs: Keyword arguments used when creating the base model in the constructor
            of `ParametrizedModel` via `from_pretrained`.
        pretrained_config: Optional config used when creating the base model in the constructor
            of `ParametrizedModel` via `from_pretrained`.

    See Also:
        `ParametrizedModelConfig`
    """

    pretrained_model_cls: Type[PreTrainedModel]
    pretrained_model_kwargs: dict[str, Any] = field(default_factory=dict)
    pretrained_config: PretrainedConfig | None = None

    def __post_init__(self):
        # if pretrained_model_cls is a string, convert it to a class (required for deserialization from JSON config)
        if isinstance(self.pretrained_model_cls, str):
            self.pretrained_model_cls = get_class_from_str(self.pretrained_model_cls)  # noqa
        else:
            self.pretrained_model_cls = self.pretrained_model_cls

    def to_dict(self) -> dict[str, Any]:
        config_dict = asdict(self)  # type: ignore
        # make sure that pretrained_model_cls and pretrained_config are JSON serializable
        config_dict["pretrained_model_cls"] = get_str_from_class(self.pretrained_model_cls)
        if self.pretrained_config is not None:
            config_dict["pretrained_config"] = self.pretrained_config.to_dict()
        return config_dict

    @classmethod
    def from_dict(cls, config_dict: dict[str, Any]) -> "BaseModelConfig":
        # try to deserialize pretrained_config with AutoConfig otherwise fall back to PretrainedConfig
        try:
            if config_dict["pretrained_config"] is not None:
                # try AutoConfig to find the right model config class
                config_dict["pretrained_config"] = AutoConfig.for_model(**config_dict["pretrained_config"])
        except ValueError:
            logger.warning("Unrecognized model identifier in AutoConfig, using PretrainedConfig instead.")
            config_dict["pretrained_config"] = PretrainedConfig.from_dict(config_dict["pretrained_config"])
        return cls(**config_dict)


# Predefined parametrization classes for `ParametrizationConfig.module_factory_cls` (avoids absolute package imports)
PARAMETRIZATION_FACTORY_REGISTRY: dict[str, Type[Parametrization]] = {
    "svd": SVDLinearParametrization,
}


@dataclass
class ParametrizationConfig:
    """
    Configuration for the parametrization to be applied to the linear layers of the base model in `ParametrizedModel`.

    Attributes:
        module_factory_cls: The class name of the parametrization to be applied to linear layers.
            Can be a string representing a class name (with absolute module path) or a predefined key
            from `PARAMETRIZATION_FACTORY_REGISTRY`.
            Use `parse_module_factory_cls` to get the actual class when creating the parametrization.
        module_factory_kwargs: Keyword arguments used when creating the parametrization with `module_factory_cls`.
        target_modules: A (list of) string(s) specifying the names of the linear layers to be parametrized.
            Follows the same semantics as Huggingface's `PeftConfig`, see also `check_target_module_exists`.
            If a string, a regex match will be performed; if a list, a module will be parametrized if its name ends
            with any of the strings in `target_modules`.
        exclude_modules: A list of strings specifying the names of the linear layers to be excluded from
            parametrization. A module will be excluded if any of the strings in `exclude_modules` is in its name.

    See Also:
        `ParametrizedModelConfig`
    """

    module_factory_cls: str
    module_factory_kwargs: dict[str, Any] = field(default_factory=dict)
    target_modules: str | list[str] | None = None
    exclude_modules: list[str] | None = None

    def parse_module_factory_cls(self) -> Type[Parametrization]:
        """Returns the class of the parametrization to be applied to linear layers."""
        try:
            if self.module_factory_cls in PARAMETRIZATION_FACTORY_REGISTRY:
                module_factory_cls = PARAMETRIZATION_FACTORY_REGISTRY[self.module_factory_cls]
            else:
                module_factory_cls = get_class_from_str(self.module_factory_cls)
        except Exception:
            raise ValueError(f"Unrecognized parametrization class: {self.module_factory_cls}")
        return module_factory_cls

    def to_dict(self) -> dict[str, Any]:
        config_dict = asdict(self)  # type: ignore
        # _maybe_include_all_linear_layers creates sets which does not work with JSON serialization, so cast to list
        for key, value in config_dict.items():
            if isinstance(value, set):
                config_dict[key] = list(value)
        return config_dict

    @classmethod
    def from_dict(cls, config_dict: dict[str, Any]) -> "ParametrizationConfig":
        return cls(**config_dict)


@dataclass
class AdapterConfig:
    """
    Configuration for the Huggingface Peft adapters to be applied to the base model.

    Attributes:
        peft_config: One or more adapter `PeftConfig`s to be applied to the base model.
            If a single `PeftConfig` is provided, it will wrapped by a dict with key "default".
            The dictionary keys will be used as adapter names in `PretrainedModel.add_adapter`.

    See Also:
        `ParametrizedModelConfig`
    """

    peft_config: PeftConfig | dict[str, PeftConfig]

    def __post_init__(self):
        if isinstance(self.peft_config, PeftConfig):
            self.peft_config = {"default": self.peft_config}

    def to_dict(self) -> dict[str, Any]:
        config_dict = asdict(self)  # type: ignore
        # Make each PeftConfig JSON serializable
        for adapter_name, peft_config in self.peft_config.items():
            peft_config_dict = peft_config.to_dict()
            # Peft casts lists to sets, which are not JSON serializable, so cast to list manually
            for key, value in peft_config_dict.items():
                if isinstance(value, set):
                    peft_config_dict[key] = list(value)
            config_dict["peft_config"][adapter_name] = peft_config_dict
        return config_dict

    @classmethod
    def from_dict(cls, config_dict: dict[str, Any]) -> "AdapterConfig":
        # Deserialize each PeftConfig automatically with from_peft_type
        for key, peft_config in config_dict["peft_config"].items():
            config_dict["peft_config"][key] = PeftConfig.from_peft_type(**peft_config)
        return cls(**config_dict)


try:
    # Prevent import errors because for some systems like macOS, bitsandbytes cannot be installed directly
    import bitsandbytes

    # Predefined quantization classes for `WeightQuantizationConfig.module_factory_cls`
    # (avoids absolute package imports)
    QUANTIZATION_FACTORY_REGISTRY: dict[str, Type[nn.Linear]] = {
        "bnb4bit": bitsandbytes.nn.Linear4bit,
    }
except ImportError:
    logger.warning("bitsandbytes is not installed, skipping quantization.")
    QUANTIZATION_FACTORY_REGISTRY: dict[str, Type[nn.Linear]] = {}


@dataclass
class WeightQuantizationConfig:
    """
    Configuration for an (optional) weight quantization to be applied to the base model.
    So far, only fp4 quantization with bitsandbytes has been tested, but analogous bitsandbytes
    quantizations should work as well. `module_factory_cls` might also use a different quantization library,
    as long as it is compatible with the module replacement strategy in `ParametrizedModule.quantize`.

    Attributes:
        module_factory_cls: The class name of the quantization to be applied to linear layers.
            Can be a string representing a class name (with absolute module path) or a predefined key
            from `QUANTIZATION_FACTORY_REGISTRY`.
            Use `parse_module_factory_cls` to get the actual class when creating the quantization.
        module_factory_kwargs: Keyword arguments used when creating the quantization with `module_factory_cls`.
        target_modules: A (list of) string(s) specifying the names of the linear layers to be quantized.
            Follows the same semantics as Huggingface's `PeftConfig`, see also `check_target_module_exists`.
            If a string, a regex match will be performed; if a list, a module will be quantized if its name ends
            with any of the strings in `target_modules`.
        exclude_modules: A list of strings specifying the names of the linear layers to be excluded from
            quantization. A module will be excluded if any of the strings in `exclude_modules` is in its name.

    See Also:
        `ParametrizedModelConfig`
    """

    module_factory_cls: str
    module_factory_kwargs: dict[str, Any] = field(default_factory=dict)
    target_modules: str | list[str] | None = None
    exclude_modules: list[str] | None = None

    def parse_module_factory_cls(self) -> Type[nn.Linear]:
        """Returns the class of the quantization to be applied to linear layers."""
        try:
            if self.module_factory_cls in QUANTIZATION_FACTORY_REGISTRY:
                module_factory_cls = QUANTIZATION_FACTORY_REGISTRY[self.module_factory_cls]
            else:
                module_factory_cls = get_class_from_str(self.module_factory_cls)
        except Exception:
            raise ValueError(f"Unrecognized quantization class: {self.module_factory_cls}")
        return module_factory_cls

    def to_dict(self) -> dict[str, Any]:
        config_dict = asdict(self)  # type: ignore
        # Make torch.dtype fields JSON serializable
        for key, value in config_dict["module_factory_kwargs"].items():
            if isinstance(value, torch.dtype):
                config_dict["module_factory_kwargs"][key] = str(value)
        # _maybe_include_all_linear_layers creates sets which does not work with JSON serialization, so cast to list
        for key, value in config_dict.items():
            if isinstance(value, set):
                config_dict[key] = list(value)
        return config_dict

    @classmethod
    def from_dict(cls, config_dict: dict[str, Any]) -> "WeightQuantizationConfig":
        # Deserialize torch.dtype fields
        for key, value in config_dict["module_factory_kwargs"].items():
            if isinstance(value, str) and value.startswith("torch."):
                dtype_name = value.split(".")[-1]
                config_dict["module_factory_kwargs"][key] = getattr(torch, dtype_name)
        return cls(**config_dict)


class ParametrizedModelConfig(PretrainedConfig):
    """
    Configuration for `ParametrizedModel` implementing a `PretrainedConfig` to be fully compatible with
    Huggingface's `PreTrainedModel` framework.

    See Also:
        - `BaseModelConfig`
        - `ParametrizationConfig`
        - `AdapterConfig`
        - `WeightQuantizationConfig`
        - `ParametrizedModel`
    """

    model_type = "parametrized_model"

    def __init__(
        self,
        base_model_config: BaseModelConfig | None = None,
        parametrization_config: ParametrizationConfig | None = None,
        adapter_config: AdapterConfig | None = None,
        weight_quantization_config: WeightQuantizationConfig | None = None,
        model_mode: Literal["train", "eval"] = "train",
        **kwargs: Any,
    ):
        """
        Initializes a `ParametrizedModelConfig`, serving as a container for `BaseModelConfig`, `ParametrizationConfig`,
        `AdapterConfig`, and `WeightQuantizationConfig`.

        Args:
            base_model_config: `BaseModelConfig`
            parametrization_config: `ParametrizationConfig`
            adapter_config: `AdapterConfig`
            weight_quantization_config: `WeightQuantizationConfig`
            model_mode: Whether to initialize the model in train or eval mode.
            **kwargs: Keyword arguments forwarded to `PretrainedConfig`.
        """
        self.base_model_config = base_model_config
        self.parametrization_config = parametrization_config
        self.adapter_config = adapter_config
        self.weight_quantization_config = weight_quantization_config
        self.model_mode = model_mode
        super().__init__(**kwargs)

    def _convert_to_dict(self, config_dict: dict[str, Any]) -> dict[str, Any]:
        if self.base_model_config is not None:
            config_dict["base_model_config"] = self.base_model_config.to_dict()
        if self.parametrization_config is not None:
            config_dict["parametrization_config"] = self.parametrization_config.to_dict()
        if self.adapter_config is not None:
            config_dict["adapter_config"] = self.adapter_config.to_dict()
        if self.weight_quantization_config is not None:
            config_dict["weight_quantization_config"] = self.weight_quantization_config.to_dict()
        return config_dict

    def to_diff_dict(self):
        # Override PretrainedConfig to_diff_dict to make subconfigs JSON serializable.
        config_dict = super().to_diff_dict()
        return self._convert_to_dict(config_dict)

    def to_dict(self):
        # Override PretrainedConfig to_diff to make subconfigs JSON serializable.
        config_dict = super().to_dict()
        return self._convert_to_dict(config_dict)

    @classmethod
    def from_dict(cls, config_dict: dict[str, Any], **kwargs: Any) -> PretrainedConfig:
        # Deserialize BaseModelConfig
        base_model_config_dict: dict[str, Any] | None = config_dict.pop("base_model_config", None)
        if base_model_config_dict is not None:
            base_model_config = BaseModelConfig.from_dict(base_model_config_dict)
        else:
            base_model_config = None
        # Deserialize ParametrizationConfig
        parametrization_config_dict: dict[str, Any] | None = config_dict.pop("parametrization_config", None)
        if parametrization_config_dict is not None:
            parametrization_config = ParametrizationConfig.from_dict(parametrization_config_dict)
        else:
            parametrization_config = None
        # Deserialize AdapterConfig
        adapter_config_dict: dict[str, Any] | None = config_dict.pop("adapter_config", None)
        if adapter_config_dict is not None:
            adapter_config = AdapterConfig.from_dict(adapter_config_dict)
        else:
            adapter_config = None
        # Deserialize WeightQuantizationConfig
        weight_quantization_config_dict: dict[str, Any] | None = config_dict.pop("weight_quantization_config", None)
        if weight_quantization_config_dict is not None:
            weight_quantization_config = WeightQuantizationConfig.from_dict(weight_quantization_config_dict)
        else:
            weight_quantization_config = None

        config = super().from_dict(config_dict, **kwargs)

        # Handle special case when return_unused_kwargs is True
        if "return_unused_kwargs" in kwargs and kwargs["return_unused_kwargs"] is True:
            config[0].base_model_config = base_model_config
            config[0].parametrization_config = parametrization_config
            config[0].adapter_config = adapter_config
            config[0].weight_quantization_config = weight_quantization_config
        else:
            config.base_model_config = base_model_config
            config.parametrization_config = parametrization_config
            config.adapter_config = adapter_config
            config.weight_quantization_config = weight_quantization_config
        return config


class ParametrizedModel(PreTrainedModel):
    """
    Base class for parametrized models implemented as a custom Huggingface `PreTrainedModel`.
    It wraps any base model of type `PreTrainedModel` in `self.model`, whose linear layers can be
    parametrized (`parametrize`), equipped with adapters (`inject_adapters`), and quantized (`quantize`).
    The corresponding modules are accessed via `parametrized_modules`, `adapter_modules`,
    and `quantized_modules`, respectively.
    The class also provides several convenience methods to manage the parametrization: `get_target_params`,
    `get_num_params`, `get_size_ratio`, `reset_target_params`, `compress`.

    Standard functionality (`forward`, `generate`, `save_pretrained`, `from_pretrained`) is essentially forwarded
    to the wrapped model.

    See Also:
        `ParametrizedModelConfig`
    """

    config_class = ParametrizedModelConfig

    def __init__(self, config: ParametrizedModelConfig, base_model: PreTrainedModel | None = None, **_: Any):
        """
        Initialize the `ParametrizedModel` from a given configuration or an existing base model.

        Args:
            config: `ParametrizedModelConfig` to be used.
            base_model: If provided, this base model is used instead of creating it from `config.base_model_config`.
            **_: Ignored keyword arguments to prevent unexpected keyword errors.

        See Also: `BaseModelConfig`
        """
        super().__init__(config)
        self.config = config  # redundant but enables type hinting for ParametrizedModelConfig

        # Either use an existing base model or create a new one from config.base_model_config
        if base_model is None:
            if self.config.base_model_config is None:
                raise ValueError("Either base_model or base_model_config must be provided.")
            self.model = self.config.base_model_config.pretrained_model_cls.from_pretrained(
                config=self.config.base_model_config.pretrained_config,
                **self.config.base_model_config.pretrained_model_kwargs,
            )
        else:
            self.model = base_model

        # Set base model to train or eval mode.
        self.train(self.config.model_mode == "train")
        logger.info(f"Base model {self.model.__class__} created.")

        # Perform parametrization.
        self._parametrized_modules: dict[str, ParametrizedModule] | None = None
        self.parametrize()

        # Inject adapters.
        self._adapter_modules: dict[str, nn.Module] | None = None
        self.inject_adapters()

        # Quantization needs to be performed manually via `quantize` because this is fully optional.
        self._quantized_modules: dict[str, nn.Linear] | None = None

        # Modified modules are initalized after parametrize and inject_adapters because they may alter the nested
        # module and parameter structure of the model.
        _ = self.parametrized_modules
        _ = self.adapter_modules
        _ = self.quantized_modules

        # Initially disable all tunable parameters to avoid unexpected behavior.
        # Tunable parameter selection should be handled by the optimizer factory in `BaseLitModule`.
        for param in self.parameters():
            param.requires_grad = False

    @property
    def base_model_name_or_path(self) -> str:
        """Convenience method to return the name or path of the base model."""
        return self.model.name_or_path  # type: ignore

    def forward(self, *args, **kwargs) -> Any:
        return self.model(*args, **kwargs)

    def generate(self, *args, **kwargs) -> Any:
        return self.model.generate(*args, **kwargs)

    def save_pretrained(
        self,
        save_directory: str | os.PathLike,
        state_dict: dict | None = None,
        include_filter: list[str] | None = None,
        exclude_filter: list[str] | None = None,
        **kwargs: Any,
    ) -> None:
        """
        Override of the default `save_pretrained` method to allow filtering of the saved state dict.

        Args:
            save_directory: Directory to save the model to.
            state_dict: Manuel override of the state dict to be saved.
                If None, `include_filter` and `exclude_filter` are applied to `self.state_dict()`.
            include_filter: List of state dict keys to include from the state dict.
                Match when the key ends with any of the strings in the list.
                If None, all keys are included.
            exclude_filter: List of state dict keys to exclude from in the state dict.
                Match when the key ends with any of the strings in the list.
                If None, no keys are excluded.
            **kwargs: Keyword arguments to be passed to the default `save_pretrained` method.

        See Also:
            `PreTrainedModel.save_pretrained`
        """
        if state_dict is None:
            state_dict = self.state_dict()
            if include_filter is not None:
                state_dict = {k: v for k, v in state_dict.items() if any(k.endswith(f) for f in include_filter)}
            if exclude_filter is not None:
                state_dict = {k: v for k, v in state_dict.items() if not any(k.endswith(f) for f in exclude_filter)}

        super().save_pretrained(save_directory=save_directory, state_dict=state_dict, **kwargs)

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str | os.PathLike | None,
        *model_args: Any,
        with_init_empty_weights: bool = True,
        **kwargs: Any,
    ) -> PreTrainedModel:
        """
        Override of the default `from_pretrained` method to allow initialization with empty weights.

        Args:
            pretrained_model_name_or_path: Model name or path.
            *model_args: Arguments to be passed to the default `from_pretrained` method.
            with_init_empty_weights: Whether to initialize the model with empty weights or not.
            **kwargs: Keyword arguments to be passed to the default `from_pretrained` method.
        """
        with init_empty_weights(with_init_empty_weights):
            return super().from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)

    @property
    def parametrized_modules(self) -> dict[str, ParametrizedModule]:
        """
        Returns a dictionary of all parametrized modules in the model.
        The returned dictionary is compatible with `self.model.named_modules()`.
        """
        if self._parametrized_modules is None:
            self._parametrized_modules = {}
            if self.config.parametrization_config is None:
                return self._parametrized_modules
            for m_name, module in self.model.named_modules():
                if isinstance(module, ParametrizedModule):
                    self._parametrized_modules[m_name] = module
        return self._parametrized_modules

    @property
    def adapter_modules(self) -> dict[str, nn.Module]:
        """
        Returns a dictionary of all adapter modules in the model.
        The returned dictionary is compatible with `self.model.named_modules()`.
        """
        if self._adapter_modules is None:
            self._adapter_modules = {}
            if self.config.adapter_config is None:
                return self._adapter_modules
            try:
                # Use the adapter management of `PreTrainedModel` to retrieve the adapter modules.
                for adapter_name in self.model.active_adapters():
                    for m_name in self.model.get_adapter_state_dict(adapter_name).keys():
                        adapter_m_name = f"{m_name.rsplit('.', 1)[0]}.{adapter_name}"
                        self._adapter_modules[adapter_m_name] = self.model.get_submodule(adapter_m_name)
            except ValueError as e:
                logger.warning(e)
        return self._adapter_modules

    @property
    def quantized_modules(self) -> dict[str, nn.Linear]:
        """
        Returns a dictionary of all quantized modules in the model.
        The returned dictionary is compatible with `self.model.named_modules()`.
        """
        if self._quantized_modules is None:
            self._quantized_modules = {}
            if self.config.weight_quantization_config is None:
                return self._quantized_modules
            try:
                module_factory_cls = self.config.weight_quantization_config.parse_module_factory_cls()
            except Exception as e:
                logger.warning(f"Could not parse weight quantization config, quantization not available.\nError: {e}")
                return self._quantized_modules
            for m_name, module in self.model.named_modules():
                if isinstance(module, module_factory_cls):
                    self._quantized_modules[m_name] = module
        return self._quantized_modules

    def parametrize(self) -> None:
        """
        Parametrize the `target_modules` from `ParametrizationConfig` using `parametrized_layer.parametrize_module`.

        See Also: `ParametrizationConfig`
        """
        if self.config.parametrization_config is None:
            logger.debug("Model parametrization is disabled.")
            return

        # Use peft semantics, e.g, "all-linear" to include all linear layers
        # TODO: Replace by own helper function to avoid unnecessary dependencies
        config: ParametrizationConfig = _maybe_include_all_linear_layers(  # type: ignore
            self.config.parametrization_config,  # type: ignore
            self.model,
        )
        module_factory_cls = config.parse_module_factory_cls()

        for m_name, module in self.model.named_modules():
            # Only modify the modules that are targeted
            if config.exclude_modules is not None and any(key in m_name for key in config.exclude_modules):
                continue
            if not check_target_module_exists(config, m_name):
                continue

            parametrization = module_factory_cls(**config.module_factory_kwargs)
            parametrize_module(module=module, parametrization=parametrization)
            logger.debug(f"Parametrized {module.__class__} module {m_name} as {parametrization.__class__}")

        self._parametrized_modules = None  # reset parametrized modules
        logger.info("Parametrization completed.")

    def inject_adapters(self) -> None:
        """
        Inject adapters according to `AdapterConfig` using the adapter management of `PreTrainedModel`.

        See Also: `AdapterConfig`
        """
        if self.config.adapter_config is None:
            logger.debug("Adapter injection is disabled.")
            return

        for adapter_name, peft_config in self.config.adapter_config.peft_config.items():
            self.model.add_adapter(peft_config, adapter_name=adapter_name)
        self.model.set_adapter(list(self.config.adapter_config.peft_config.keys()))

        self._adapter_modules = None  # reset adapter modules
        logger.info("Adapters injected.")

    def quantize(self) -> None:
        """
        Quantize the `target_modules` from `WeightQuantizationConfig`.

        See Also: `WeightQuantizationConfig`
        """
        if self.config.weight_quantization_config is None:
            logger.debug("Weight quantization is disabled.")
            return

        # Use peft semantics e.g "all-linear" to include all linear layers
        # TODO: Replace by own helper function to avoid unnecessary dependencies
        config: WeightQuantizationConfig = _maybe_include_all_linear_layers(  # type: ignore
            self.config.weight_quantization_config,  # type: ignore
            self.model,
        )
        module_factory_cls = config.parse_module_factory_cls()

        for m_name, module in self.model.named_modules():
            # Only modify the modules that are targeted
            if config.exclude_modules is not None and any(key in m_name for key in config.exclude_modules):
                continue
            if not check_target_module_exists(config, m_name) or isinstance(module, ParametrizedModule):
                continue
            if not isinstance(module, nn.Linear):
                continue

            # Important: This module must NOT be created in a device context like with_init_device("cuda")
            quantized_module = module_factory_cls(
                module.in_features,
                module.out_features,
                bias=module.bias is not None,
                device=module.weight.device,
                **config.module_factory_kwargs,
            )
            # cf. https://huggingface.co/docs/bitsandbytes/reference/nn/linear4bit#bitsandbytes.nn.Linear4bit.example
            quantized_module.load_state_dict(module.state_dict())
            quantized_module = quantized_module.to(module.weight.device)
            quantized_module.weight.requires_grad = False
            logger.debug(f"Quantized {module.__class__} module {m_name} to {quantized_module.__class__}")

            # Replace the target module by the quantized module
            parent_name, child_name = m_name.rsplit(".", 1)
            parent_module = self.model.get_submodule(parent_name)
            parent_module.add_module(child_name, quantized_module)

        self._quantized_modules = None  # reset quantized modules
        logger.info("Quantization completed.")

    def get_target_params(self) -> dict[str, nn.Parameter]:
        """
        Lifts `Parametrization.get_target_params` to the model scope.
        The returned dictionary should be compatible with `self.model.named_parameters()`.

        See Also:
            `Parametrization.get_target_params`
        """
        target_params = {}
        for m_name, module in self.parametrized_modules.items():
            for p_name, param in module.parametrization.get_target_params().items():
                target_params[f"{m_name}.parametrization.{p_name}"] = param
        return target_params

    def get_num_params(
        self, compressed: bool = False, full: bool = False, target_params: dict[str, torch.Tensor] | None = None
    ) -> int:
        """
        Lifts `Parametrization.get_num_params` to the model scope.
        Computes the (effective) number of parameters of the entire model.

        Args:
            compressed: Whether to count the number of parameters as if the parametrized modules were actually
                compressed. If `False`, the number of parameters is the same as in the original module.
            full: If `True`, all parameters of the model are counted, if `False` only those of parametrized modules.
                Default is `False`, which follows the most common convention in the compression literature.
            target_params: Count the number of parameters as if `target_params` were used instead of
                the parametrized modules' target parameters. The dictionary keys should be compatible with those of
                `self.get_target_params`.

        See Also:
            `Parametrization.get_num_params`
        """
        num_params_full = 0
        if full:
            for name, param in self.model.named_parameters():
                if "parametrization" not in name:  # exclude parametrized modules here (counted below)
                    if hasattr(param, "quant_state"):  # HOTFIX: special case for bitsandbytes-quantized parameters
                        num_params_full += param.numel() * 2
                    else:
                        num_params_full += param.numel()

        num_params = 0
        for module_name, module in self.parametrized_modules.items():
            module_target_params = None
            if compressed and target_params is not None:
                # Make target_params' keys those of parametrized models, i.e., trim f"{module_name}.parametrization."
                prefix = f"{module_name}.parametrization."
                # Filter and re-map keys for the current module
                module_target_params = {
                    key[len(prefix) :]: value for key, value in target_params.items() if key.startswith(prefix)
                }
                if not module_target_params:
                    module_target_params = None

            num_params += module.parametrization.get_num_params(
                compressed=compressed, target_params=module_target_params
            )
        num_params = num_params + num_params_full
        if num_params == 0:
            # dummy to avoid division by zero (e.g., if there are no parametrized_modules and full=False)
            num_params = 1e-6
        return num_params

    def get_size_ratio(self, full: bool = False, target_params: dict[str, torch.Tensor] | None = None) -> float:
        """
        Convenience function to compute the size ratio of the present model.

        See Also:
            `get_num_params`
        """
        return self.get_num_params(compressed=True, full=full, target_params=target_params) / self.get_num_params(
            full=full
        )

    def reset_target_params(self, mode: Literal["full", "nonzero", "compress"] = "full") -> None:
        """
        Lifts `Parametrization.reset_target_params` to the model scope.

        Args:
            mode: The reset mode, see `Parametrization.reset_target_params`.

        See Also:
            `Parametrization.reset_target_params`
        """
        for m_name, module in self.parametrized_modules.items():
            module.parametrization.reset_target_params(mode=mode)

    def compress(self) -> None:
        """
        Compresses all parametrized modules using `Parametrization.reset_target_params(mode="compress")`.
        If no compression is possible, the module is unparametrized and removed from `parametrized_modules`.
        """
        removed_parametrized_modules = []
        for m_name, module in self.parametrized_modules.items():
            if module.parametrization.get_num_params(compressed=True) / module.parametrization.get_num_params() >= 1.0:
                unparametrize_module(module)
                removed_parametrized_modules.append(m_name)
                logger.debug(f"Unparametrizing {module.__class__} module {m_name}")
            else:
                module.parametrization.reset_target_params(mode="compress")
                logger.debug(f"Compressing {module.__class__} module {m_name}")
        for m_name in removed_parametrized_modules:
            self.parametrized_modules.pop(m_name)
        logger.info("Compression completed.")


# Register ParametrizedModelConfig and ParametrizedModel for AutoModel
# Required to push custom model to Huggingface Hub (see https://huggingface.co/docs/transformers/en/custom_models)
ParametrizedModelConfig.register_for_auto_class()
ParametrizedModel.register_for_auto_class("AutoModel")
