from abc import ABC, abstractmethod
from typing import ClassVar, Literal, Protocol, runtime_checkable, Type

import torch
from torch import nn


class Parametrization(nn.Module, ABC):
    """
    Abstract base class for parametrizations.
    A parametrization can be injected into any torch module of type `base_class` by `parametrize_module`.
    A parametrized module will follow the `ParametrizedModule` interface.

    This will overload the weight, bias, and forward of the module so that they play together with
    the parametrization. The external behavior of the parametrized module remains unchanged, for instance,
    a parametrized `Linear` module will still work as expected.

    Attributes:
        base_class: The base class of the module that can be parametrized.
        initialized: A flag that indicates whether the parametrization has been initialized.
    """

    initialized: bool = False
    base_class: ClassVar[Type[nn.Module]]

    def initialize(self, base_module: "Parametrization.base_class") -> None:
        self._initialize(base_module)
        self.initialized = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute the forward pass of the parametrization.
        This is particularly important when a standard forward pass based on `weight` would be inefficient.
        """
        assert self.initialized
        x = self._forward(x)
        return x

    @property
    def weight(self) -> torch.Tensor:
        """Compute the weight tensor of the parametrization."""
        return self._weight()

    @property
    def bias(self) -> torch.Tensor | None:
        """Compute the bias tensor of the parametrization."""
        return self._bias()

    @abstractmethod
    def _forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    @abstractmethod
    def _initialize(self, base_module: "Parametrization.base_class") -> None:
        """
        Initialize the parametrization based on a given base module.
        This method should build the internal representation the module's weight and bias,
        registering all required buffers and parameters in `self`.
        """
        raise NotImplementedError

    @abstractmethod
    def _weight(self) -> torch.Tensor:
        raise NotImplementedError

    @abstractmethod
    def _bias(self) -> torch.Tensor | None:
        raise NotImplementedError

    @abstractmethod
    def get_target_params(self) -> dict[str, torch.nn.Parameter]:
        """
        Return the (tunable) target parameters of the parametrization.
        Here, "target parameters" means that they can be tuned and potentially compressed
        by `self.reset_target_params(mode="compress")`.
        Other torch parameters of the module could be tuned as well, but should not returned here.
        The returned dictionary should be compatible with `self.named_parameters()`.

        See Also:
            - `ParametrizedModel.get_target_params`
            - `ParametrizedModel.compress`
        """
        raise NotImplementedError

    @abstractmethod
    def reset_target_params(self, mode: Literal["full", "nonzero", "compress"] = "full") -> None:
        """
        Reset the target parameters of the parametrization according to a given mode.

        Args:
            mode: The reset mode.
                "full" means reset to original value at initialization.
                "nonzero" means reset all non-zero values to original value at initialization.
                "compress" means the all zero values are removed and the the parameters are compressed accordingly.
        """
        raise NotImplementedError

    @abstractmethod
    def get_num_params(self, compressed: bool = False, target_params: dict[str, torch.Tensor] | None = None) -> int:
        """
        Computes the (effective) number of parameters of the parametrization.

        Args:
            compressed: Whether to count the number of parameters as if the module was actually compressed.
                If `False`, the number of parameters is the same as in the original module.
            target_params: Count the number of parameters as if `target_params` were used instead of
                `self.get_target_params()`. This "what if" feature is important when pruning
                a full `ParametrizedModel` to a certain target ratio.
        """
        raise NotImplementedError


@runtime_checkable
class ParametrizedModule(Protocol):
    """
    Interface for a parametrized `nn.Module`.
    It ensures that `weight` and `bias` are forwarded to the `Parametrization` instance.

    Attributes:
        parametrization: The `Parametrization` instance of the module.
        _forward: The original forward function of the module.
        __old_class__: The original class of the module.

    Notes:
        `_forward` and `__old_class__` are used by `parametrize_module` and `unparametrize_module`
         to allow restoring the original behavior of the module.
    """

    parametrization: Parametrization
    _forward: callable
    __old_class__: type[nn.Module]

    @property
    def weight(self):
        return self.parametrization.weight

    @property
    def bias(self):
        return self.parametrization.bias


def parametrize_module(module: nn.Module, parametrization: Parametrization) -> ParametrizedModule and nn.Module:
    """
    Parametrize a module using a `Parametrization` instance.

    Args:
        module: The module to be parametrized.
        parametrization: The `Parametrization` instance to be applied to the module.

    Returns: The parametrized module using the `ParametrizedModule` interface.

    Notes:
        Adopted from https://stackoverflow.com/a/31075641
    """

    assert isinstance(module, parametrization.base_class)
    module.__old_class__ = module.__class__

    # Initializes the parametrization and adds it to the module
    module.add_module("parametrization", parametrization)
    module.parametrization.initialize(module)

    # Save the original forward in case we want to remove the parametrization again
    module._forward = module.forward

    # Cast to new parametrized object class type
    del module.weight
    del module.bias
    module.__class__ = type("Parametrized" + module.__class__.__name__, (module.__class__, ParametrizedModule), {})
    # Make sure that we utilize the forward function of the parametrization
    module.forward = module.parametrization.forward

    return module


def unparametrize_module(module: ParametrizedModule) -> nn.Module:
    """
    Revert the parametrization of a module.

    Args:
        module: A module that has been parametrized by `parametrize_module`.

    Returns: The original module.

    Notes:
        Adopted from https://stackoverflow.com/a/31075641
    """

    # Make sure to save weight and bias in intermediate variables
    weight = module.weight
    bias = module.bias

    assert isinstance(module, nn.Module)

    # This line will remove properties module.weight and module.bias
    module.__class__ = type(module.__old_class__.__name__, (module.__old_class__,), {})
    delattr(module, "__old_class__")

    # Add weight and bias as native parameters to the module again
    module.register_parameter("weight", nn.Parameter(weight, weight.requires_grad))
    if bias is not None:
        module.register_parameter("bias", nn.Parameter(bias, bias.requires_grad))
    else:
        module.register_parameter("bias", None)

    # Recover the original forward pass and get rid of the parametrization
    del module.parametrization
    module.forward = module._forward
    delattr(module, "_forward")

    return module
