from abc import ABC, abstractmethod
from collections import defaultdict
from logging import getLogger
from typing import Any, Type

import torch

from acip.core.utils import get_class_from_str

logger = getLogger(__name__)


class OptimizerFactory(ABC):
    """
    Factory for a creating PyTorch optimizer. Used in the `configure_optimizers` hook of `BaseLitModule`.
    """

    @abstractmethod
    def create_optimizer(self, model: torch.nn.Module) -> torch.optim.Optimizer:
        """Create the optimizer."""
        raise NotImplementedError

    def __call__(self, model: torch.nn.Module) -> torch.optim.Optimizer:
        """Create the optimizer."""
        return self.create_optimizer(model)


class BaseOptimizerFactory(OptimizerFactory):
    """
    Implements a general optimizer factory for a given optimizer class.
    It allows specifying parameter groups and individual optimizer arguments for each group.
    """

    def __init__(
        self,
        optimizer_cls: Type[torch.optim.Optimizer] = torch.optim.AdamW,
        normal_params_kwargs: dict[str, Any] | None = None,
        group_params: dict[str, list[str]] | None = None,
        group_params_kwargs: dict[str, dict[str, Any]] | None = None,
        general_optimizer_kwargs: dict[str, Any] | None = None,
        reset_tunable_params: bool = False,
    ):
        """
        Args:
            optimizer_cls: The PyTorch optimizer class to be used.
            normal_params_kwargs: Optimizer arguments for all parameters with `requires_grad=True`,
                which are not assigned to any group in `group_params`.
            group_params: Dictionary specifying parameter groups for the optimizer. Keys are group names and values are
                lists of parameter names to be included. Parameter names are matched if they end with one of the
                strings in the list.
            group_params_kwargs: Optional optimizer arguments for each group in `group_params`.
                Keys are group names and values contain the optimizer arguments for each group.
            general_optimizer_kwargs: General optimizer arguments applied to normal parameters and all groups.
            reset_tunable_params: If True, all model parameters are set `requires_grad=False` initially when
                creating the optimizer. This implies that there are no normal parameters and all tunable parameters
                are managed by the groups in `group_params` (their `requires_grad` flag is set to True again).
                This flag is useful when you want make sure that there are no unexpected tunable parameters
                in the model.
        """
        if isinstance(optimizer_cls, str):
            self.optimizer_cls = get_class_from_str(optimizer_cls)
        else:
            self.optimizer_cls = optimizer_cls
        self.normal_params_kwargs = normal_params_kwargs if normal_params_kwargs is not None else {}
        self.group_params = group_params if group_params is not None else {}
        self.group_params_kwargs = group_params_kwargs if group_params_kwargs is not None else defaultdict(dict)
        self.general_optimizer_kwargs = general_optimizer_kwargs if general_optimizer_kwargs is not None else {}
        self.reset_tunable_params = reset_tunable_params

        # Store references to the tunable parameters.
        # This is useful when certain parameters should be set to requires_grad=False or True later in training.
        # The dict keys are compatible with the group names.
        self.params_normal: dict[str, torch.nn.Parameter] | None = None
        self.params_grouped: dict[str, dict[str, torch.nn.Parameter]] | None = None

    def create_optimizer(self, model: torch.nn.Module) -> torch.optim.Optimizer:
        if self.reset_tunable_params:
            for param in model.parameters():
                param.requires_grad = False

        # Initially collect all tunable parameters in params_normal
        params_normal = {p_name: param for p_name, param in model.named_parameters() if param.requires_grad}
        # Build parameter groups based on group_params
        params_grouped = defaultdict(dict)
        for p_name, param in model.named_parameters():
            for group in self.group_params:
                for group_p_name in self.group_params[group]:
                    if p_name.endswith(group_p_name):
                        # Explicitly set requires_grad to True
                        param.requires_grad = True
                        params_grouped[group][p_name] = param
                        if p_name in params_normal:
                            # Make sure that the parameter is not in normal group anymore
                            del params_normal[p_name]
        # Make parameter groups compatible with the optimizer class input arguments
        optimizer_groups = []
        if len(params_normal) > 0:
            optimizer_groups.append(dict(params=list(params_normal.values()), **self.normal_params_kwargs))
            for p_name in params_normal:
                logger.debug(f"Optimizer - normal param: {p_name}")
        if len(params_grouped) > 0:
            for group, params in params_grouped.items():
                if len(params) > 0:
                    optimizer_groups.append(dict(params=list(params.values()), **self.group_params_kwargs[group]))
                for p_name in params:
                    logger.debug(f"Optimizer - group {group} param: {p_name}")
        self.params_normal = params_normal
        self.params_grouped = params_grouped
        return self.optimizer_cls(optimizer_groups, **self.general_optimizer_kwargs)
