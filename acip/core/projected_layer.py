import math
from abc import ABC, abstractmethod
from logging import getLogger
from typing import Literal

import torch
from torch import nn
from torch.nn import functional as F

from .parametrized_layer import Parametrization
from .utils import use_init_empty_weights

logger = getLogger(__name__)


class CompressionCriterion(ABC):
    """
    Abstract class for compression criterion of a (target) parameter of a parametrized module.
    """

    @abstractmethod
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: A tensor of any shape

        Returns: A boolean mask of the same shape as `x` where `False` indicates that the entry can be removed.
        """
        raise NotImplementedError


class ThresholdCriterion(CompressionCriterion):
    """
    Compression criterion based on a threshold. All entries below `self.threshold` can be removed.
    """

    def __init__(self, threshold: float = 0.0):
        self.threshold = threshold

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return x > self.threshold


class ProjectedLinearParametrization(Parametrization, ABC):
    """
    Implementation of a linear layer parametrization, factorizing the weight matrix as
    `weight = ortho.weight @ torch.diag(mask) @ base.weight`.
    Here, `ortho` is a linear layer with orthogonal columns, `mask` represents a (binary) diagonal matrix
    that can be pruned, and `base` is a linear layer (determined by the choice of `ortho`).
    Any child class needs to implement `_ortho_init` which creates `ortho`. Based on this, `mask` and `base` are
    initialized such that the original weight matrix is obtained at initialization.

    `mask` corresponds to the only target parameter of this parametrization. Pruning it will result in
    a low-rank matrix representation of the parametrized linear module.
    """

    base_class = nn.Linear

    def __init__(
        self,
        mask_func: Literal["ste", "relu", "none"] = "ste",
        mask_scaling_factor: float | str = "norm",
        compression_criterion: CompressionCriterion = ThresholdCriterion(),
    ):
        """
        Args:
            mask_func: A function applied to the mask parameter in each forward pass implementing
                custom functionalities. Available options: ["ste", "relu", "none"].
                "ste" means using a straight-through estimator, i.e., in the forward pass, `mask` is binarized, which
                is ignored in the backward pass. Before `mask` passed through a ReLU activation.
                "relu" means that `mask` is passed through a ReLU activation.
                "none" means that `mask` is not modified.
            mask_scaling_factor: Conceptually, `mask` is initialized with ones, but rescaling to a smaller value
                can vastly improve the training speed. `mask_scaling_factor` specifies this rescaling factor.
                The rescaling should be compensated by scaling `ortho` accordingly in `self._ortho_init`.
                If `mask_scaling_factor='norm'`, the scaling factor is chosen such that `mask` has unit L2 norm
                (note that this can lead to a different behavior in model tuning than for a fixed factor
                 when some target parameters have different number of elements).
            compression_criterion: `CompressionCriterion` to be used in `self.reset_target_params(mode="compress")`.
        """
        super().__init__()
        self.mask_func = {
            "ste": mask_func_ste,
            "relu": mask_func_relu,
            "none": mask_func_none,
        }[mask_func]
        self._mask_scaling_factor = mask_scaling_factor
        self.compression_criterion = compression_criterion

    def _forward(self, x: torch.Tensor) -> torch.Tensor:
        # This implementation avoids an explicit materalization of `weight`.
        x = self.base(x)
        x = self.mask_func(self.mask, self.mask_scaling_factor) * x
        x = self.ortho(x)
        return x

    def _weight(self) -> torch.Tensor:
        # Compute the original weight matrix, don't use this in forward pass for efficiency reasons
        mask = self.mask_func(self.mask, self.mask_scaling_factor)
        return self.ortho.weight @ torch.diag(mask) @ self.base.weight

    def _bias(self) -> torch.Tensor | None:
        return self.ortho.bias

    def _initialize(self, base_module: base_class) -> None:
        factory_kwargs = {"device": base_module.weight.device, "dtype": base_module.weight.dtype}
        in_dim, out_dim = base_module.in_features, base_module.out_features
        proj_dim = min(in_dim, out_dim)  # infer mask (bottleneck) dimension

        # Initialize ortho layer ....
        self.add_module(
            "ortho",
            nn.Linear(in_features=proj_dim, out_features=out_dim, bias=base_module.bias is not None, **factory_kwargs),
        )
        self._ortho_init(base_module.weight)
        if base_module.bias is not None:
            # It is important that ortho carries the bias (and not base) because ortho is used to compute the final
            # output of the forward pass
            self.ortho.bias.data.copy_(base_module.bias.data)

        # ... and compute the base layer based on the choice of ortho (this only works of ortho has orthogonal columns)
        base = base_module.__class__(in_features=in_dim, out_features=proj_dim, bias=False, **factory_kwargs)
        base.weight.data.copy_(self.ortho.weight.data.T @ base_module.weight.data)
        self.add_module("base", base)

        # Creating (tunable) mask parameter ...
        self.register_parameter("mask", torch.nn.Parameter(torch.ones(proj_dim, **factory_kwargs)))
        # ... and rescale mask properly in a separate step
        # (because reset_target_params calls mask_scaling_factor, which in turn may require mask to already exist)
        self.reset_target_params()

    @abstractmethod
    def _ortho_init(self, weight: torch.Tensor) -> None:
        """
        Initialize ortho layer. Must be implemented by child class.

        Args:
            weight: Weight matrix of the original linear layer module.
        """
        raise NotImplementedError

    def get_target_params(self) -> dict[str, torch.nn.Parameter]:
        return {"mask": self.mask}

    @property
    def mask_scaling_factor(self) -> float:
        if self._mask_scaling_factor == "norm":
            # Choose scaling factor such that mask has unit L2 norm.
            # Note: mask already needs to exist at this point to infer its shape.
            self._mask_scaling_factor = 1 / math.sqrt(self.mask.numel())
            return self._mask_scaling_factor
        elif isinstance(self._mask_scaling_factor, float):
            return self._mask_scaling_factor
        else:
            raise ValueError(f"Invalid mask_scaling_factor: {self._mask_scaling_factor}")

    @property
    def in_features(self) -> int:
        return self.base.in_features

    @property
    def out_features(self) -> int:
        return self.ortho.out_features

    def reset_target_params(self, mode: Literal["full", "nonzero", "compress"] = "full") -> None:
        with torch.no_grad():
            if mode == "full":
                # Scale mask values properly by self.mask_scaling_factor
                self.mask.data = torch.ones_like(self.mask.data) * self.mask_scaling_factor
            elif mode == "nonzero":
                # Scale mask values properly by self.mask_scaling_factor
                self.mask.data[self.mask.data > 0] = 1.0 * self.mask_scaling_factor
                self.mask.data[self.mask.data < 0] = 0.0
            elif mode == "compress":
                if self.compression_criterion is None:
                    logger.warning("Compression criterion is not set. No op...")
                    return
                # Select entries of parameter mask that should be kept
                dim_select = self.compression_criterion(self.mask)
                # Create and register compressed layers and mask
                new_base = new_linear_from_mask(self.base, dim_select, column_select=False)
                new_ortho = new_linear_from_mask(self.ortho, dim_select, column_select=True)
                new_mask = self.mask[dim_select].clone().detach()
                del self.mask, self.base, self.ortho
                self.register_module("base", new_base)
                self.register_module("ortho", new_ortho)
                self.register_parameter("mask", nn.Parameter(new_mask))
            else:
                raise ValueError(f"Invalid mode: {mode}")

    def get_num_params(self, compressed: bool = False, target_params: dict[str, torch.Tensor] | None = None) -> int:
        if not compressed:
            # Compute number of parameters for full linear layer
            num_params = self.in_features * self.out_features
            if self.bias is not None:
                num_params += self.out_features
            return num_params
        else:
            # Compute number of mask values that could be discarded by self.reset_target_params(mode="compress") ...
            if target_params is not None:
                sparsity = mask_sparsity(target_params["mask"] != 0.0, threshold=0.0)
            else:
                sparsity = mask_sparsity(self.mask)
            # ... and compute the (hypothetical) number of parameters for a compressed module.
            num_params = self.in_features * sparsity + sparsity * self.out_features
            if self.bias is not None:
                num_params += self.out_features
            # If the number of parameters for the compressed module would be larger than the number of parameters
            # for the full module, return the latter because we can always unparametrize to the original module if
            # compression would not be effective.
            num_params = min(self.get_num_params(compressed=False), num_params)
            return num_params


class SVDLinearParametrization(ProjectedLinearParametrization):
    """
    Implementation of a linear layer parametrization using SVD decomposition.
    If the SVD of weight is U * S * V^T, then `ortho.weight = U` and `base.weight = S * V^T`.
    As base is computed automatically by `_initialize`, `_ortho_init` only needs to compute U and
    scale it properly with `mask_scaling_factor`. The singular values S are buffered just in case they are needed
    in the tuning process.
    """

    def _ortho_init(self, weight: torch.Tensor) -> None:
        k = min(weight.shape[0], weight.shape[1])
        if use_init_empty_weights.get():
            # Check if the init_empty_weights context is active which avoids a (costly) SVD computation and just
            # initializes U and S as empty tensors. They are loaded later from a pretrained model.
            logger.debug("Parametrizing with empty weights.")
            U = torch.empty(weight.shape[0], k)
            S = torch.empty(k, 1)
        else:
            # Detaching is important to avoid memory leaks. torch.linalg.svd only works with float32.
            U, S, _ = torch.linalg.svd(weight.detach().float(), full_matrices=False)
            # Rescaling U based on mask_scaling_factor
            # This step is somewhat manual because calling mask_scaling_factor requires the mask to already exist
            if self._mask_scaling_factor == "norm":
                U = math.pow(k, 1 / 4) * U
            else:
                U = math.sqrt(1 / self._mask_scaling_factor) * U
        factory_kwargs = {"device": weight.device, "dtype": weight.dtype}
        self.ortho.weight.data.copy_(U.detach().to(**factory_kwargs))
        self.register_buffer("S", S.detach().flatten().to(**factory_kwargs))


def mask_func_ste(mask: torch.Tensor, mask_scaling_factor: float) -> torch.Tensor:
    # See ProjectedLinearParametrization.__init__ for more details.
    mask = F.relu(mask)
    return (mask > 0).to(mask.dtype).detach() * mask_scaling_factor + mask - mask.detach()


def mask_func_relu(mask: torch.Tensor, mask_scaling_factor: float) -> torch.Tensor:
    # See ProjectedLinearParametrization.__init__ for more details.
    return F.relu(mask)


def mask_func_none(mask: torch.Tensor, mask_scaling_factor: float) -> torch.Tensor:
    # See ProjectedLinearParametrization.__init__ for more details.
    return mask


def mask_sparsity(mask: torch.Tensor, threshold: float = 0.0) -> int:
    """Simple util function to compute the number of non-zero elements of a mask, where an element is considered
    non-zero if its value is strictly greater than `threshold`."""
    return torch.count_nonzero(mask > threshold).item()


def new_linear_from_mask(module: nn.Linear, dim_select: torch.Tensor, column_select=True) -> nn.Linear:
    """
    Creates a new linear layer from an existing one based on a mask indicating which columns/rows to keep.

    Args:
        module: Module to be pruned.
        dim_select: Boolean tensor mask indicating which columns/rows to keep.
        column_select: Whether to prune columns (True) or rows (False) according to `dim_select`.

    Returns: Pruned module.
    """
    assert dim_select.dtype == torch.bool, "dim_select must be boolean"

    in_features, out_features = module.in_features, module.out_features
    sparsity = dim_select.sum().item()
    if column_select:
        in_features = sparsity
    else:
        out_features = sparsity
    new_module = module.__class__(
        in_features=in_features,
        out_features=out_features,
        bias=module.bias is not None,
        device=module.weight.device,
        dtype=module.weight.dtype,
    )
    weight = module.weight.data
    if column_select:
        weight = weight[:, dim_select]
    else:
        weight = weight[dim_select, :]
    new_module.weight.data.copy_(weight.detach())

    if new_module.bias is not None:
        if column_select:
            new_module.bias.data.copy_(module.bias.detach())
        else:
            # If rows are pruned, the bias needs to be pruned as well
            new_module.bias.data.copy_(module.bias[dim_select].detach())

    return new_module
