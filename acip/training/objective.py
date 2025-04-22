from abc import ABC, abstractmethod
from typing import Any

import torch
from torch import nn
from transformers import GenerationMixin, PreTrainedModel
from transformers.modeling_outputs import CausalLMOutput

from acip.core.parametrized_model import ParametrizedModel


class Objective(nn.Module, ABC):
    """
    Base class for training objectives used by `BaseLitModule`.
    The forward pass is supposed to take a batch and a model as input and return a dict of metrics.
    The returned dict should contain at least a "loss" key used as key objective for training.
    Moreover, it may contain additional metrics with an "info_" prefix that are logged by `BaseLitModule`.
    It is also useful to return the loss itself as a separate info metric when the objective can be combined
    with others by `CombinedLoss`.
    """

    @abstractmethod
    def forward(self, batch: dict[str, Any], model: PreTrainedModel) -> dict[str, Any]:
        raise NotImplementedError


class TaskLoss(Objective):
    """
    Simple task loss for causal language modeling.
    Info metrics: loss and perplexity
    """

    def forward(self, batch: dict[str, Any], model: PreTrainedModel) -> dict[str, Any]:
        assert isinstance(model, GenerationMixin)
        model_output: CausalLMOutput = model(
            batch["input_ids"],
            labels=batch["labels"],
            attention_mask=batch["attention_mask"],
        )
        task_loss = model_output.loss
        ppl = torch.exp(task_loss.detach())
        return {"loss": task_loss, "info_task_loss": task_loss.item(), "info_ppl": ppl.item()}


class LpRegularizationLoss(Objective):
    """
    L_p-norm regularization loss to penalize the target parameters of a `ParametrizedModel`.
    The returned loss is the sum of the L_p-norms of the target parameters.
    Info metrics: loss

    Notes: This loss will only work properly if the target parameters are non-negative.
        We use a "one-sided" L1 norm to keep already pruned parameters zero, i.e., a ReLU "kills" any signal
        of parameters that are already zero. Omitting this step can lead to unexpected training dynamics.
    """

    def __init__(
        self,
        reg_lp_norm: float = 1.0,
        reg_precision: str | torch.dtype = torch.float32,
    ):
        """
        Args:
            reg_lp_norm: Which L_p-norm to use.
            reg_precision: The target parameters are cast to this precision before computing the L_p-norm.
                float32 is recommended in most cases.
        """
        super().__init__()
        self.reg_lp_norm = reg_lp_norm
        if isinstance(reg_precision, str):
            reg_precision = getattr(torch, reg_precision)
        self.reg_precision = reg_precision

    def forward(self, batch: dict[str, Any], model: PreTrainedModel) -> dict[str, Any]:
        assert isinstance(model, ParametrizedModel)
        lp_norms = {}
        for p_name, param in model.get_target_params().items():
            # "One-sided" L1 norm to keep already pruned parameters zero
            lp_norms[p_name] = torch.norm(nn.functional.relu(param.to(self.reg_precision)), p=self.reg_lp_norm)
        reg_loss = sum([lp_norm for lp_norm in lp_norms.values()])
        return {
            "loss": reg_loss,
            # if reg_loss is not a tensor, lp_norms was empty
            "info_reg_loss": reg_loss.item() if isinstance(reg_loss, torch.Tensor) else 0.0,
        }


class CombinedLoss(Objective):
    """
    A loss objective that computes a weighted sum of other loss objectives.
    This is particularly useful for regularized task loss functions.
    """

    def __init__(self, loss_objectives: dict[str, Objective], loss_weights: dict[str, float]):
        """
        Args:
            loss_objectives: Dictionary of loss objectives.
            loss_weights: Weights for each loss objective. The keys must be the same as in `loss_objectives`.
        """
        super().__init__()
        assert set(loss_objectives.keys()) == set(loss_weights.keys())
        self.loss_objectives = loss_objectives
        self.loss_weights = loss_weights

    def forward(self, batch: dict[str, Any], model: PreTrainedModel) -> dict[str, Any]:
        output = {}
        total_loss = 0.0
        for key in self.loss_objectives.keys():
            output_objective = self.loss_objectives[key](batch, model)
            total_loss += self.loss_weights[key] * output_objective["loss"]
            # Forward info metrics to output
            output.update(output_objective)
        output["loss"] = total_loss
        return output
