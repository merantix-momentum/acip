from abc import ABC
from logging import getLogger
from typing import Any

import numpy as np
import torch
import wandb
from lightning import pytorch as pl
from lightning.pytorch.utilities.types import STEP_OUTPUT

from acip.core.acip_model import ACIPModel
from acip.core.parametrized_model import ParametrizedModel
from acip.core.projected_layer import SVDLinearParametrization
from acip.training.objective import CombinedLoss
from acip.training.optimizer import BaseOptimizerFactory
from acip.training.pl_module import BaseLitModule
from acip.training.utils import generate_params_plot, stop_train

logger = getLogger(__name__)


def size_ratio_reached(pl_module: BaseLitModule, size_ratio: float, full: bool = False) -> bool:
    """
    Helper function to check if a given model has reached a size ratio.
    Serves as standard stopping criterion for ACIP.
    """
    assert isinstance(pl_module.model, ParametrizedModel)
    return size_ratio >= pl_module.model.get_size_ratio(full=full)


class ACIPScheduler(pl.Callback):
    """
    Lightning callback that schedules the ACIP tuning process.
    It manages the scheduling of the regularization parameter, stopping criterion, and transition to
    an optional post-tuning phase.
    Under the hood, the training phases are managed with the `BaseLitModule.training_phase` attribute.
    """

    def __init__(
        self,
        acip_stop_ratio: float,
        measure_ratio_full: bool = False,
        post_tune_steps: int | None = None,
        reg_scheduler_start_weight: float = 0.001,
        reg_scheduler_update_every: int | None = None,
        reg_scheduler_update_factor: float = 2.0,
    ):
        """
        Args:
            acip_stop_ratio: Size ratio at which to stop ACIP.
            measure_ratio_full: If `True`, all parameters of the model are counted when computing
                the stopping criterion, if `False` only the parameters of the parametrized modules
                are counted (default). See `full` flag in `ParametrizedModel.get_size_ratio`.
            post_tune_steps: Optional number of tuning steps to run after the ACIP stopping criterion is reached.
                This will continue optimizing the "tuning_params" defined in the optimizer with respect to the
                task objective. None means post-tuning is disabled.
            reg_scheduler_start_weight: Initial value of the regularization parameter
                (= initial loss weight of "reg_objective" in CombinedLoss).
            reg_scheduler_update_every: If an integer, the regularization parameter is scaled by
                `reg_scheduler_update_factor` every `reg_scheduler_update_every` steps.
                None means regularization parameter updates are disabled.
            reg_scheduler_update_factor: The factor by which to scale the regularization parameter.
        """
        super().__init__()
        self.acip_stop_ratio = acip_stop_ratio
        self.measure_ratio_full = measure_ratio_full
        self.post_tune_steps = post_tune_steps
        self.reg_scheduler_start_weight = reg_scheduler_start_weight
        self.reg_scheduler_update_every = reg_scheduler_update_every
        self.reg_scheduler_update_factor = reg_scheduler_update_factor

        # Internal counters for the number of completed ACIP and post-tuning steps, respectively.
        self._acip_step = 0
        self._post_tune_step = 0

    def on_fit_start(self, trainer: pl.Trainer, pl_module: BaseLitModule) -> None:
        # Assign initial value to the regularization parameter.
        assert isinstance(pl_module.objective, CombinedLoss)
        assert "reg_objective" in pl_module.objective.loss_weights
        pl_module.objective.loss_weights["reg_objective"] = self.reg_scheduler_start_weight

        # Initialize the ACIP phase.
        pl_module.training_phase = "acip"

    def on_train_batch_end(
        self, trainer: pl.Trainer, pl_module: BaseLitModule, outputs: STEP_OUTPUT, batch: Any, batch_idx: int
    ) -> None:
        if pl_module.training_phase == "acip":
            self._acip_step += 1
            self._update_reg_weight(pl_module)

            # If stopping criterion is reached, transition to the post-tuning phase or stop training directly.
            if size_ratio_reached(pl_module, self.acip_stop_ratio, full=self.measure_ratio_full):
                if self.post_tune_steps is None:
                    stop_train(trainer)
                else:
                    self._prepare_post_tune(pl_module)
                    pl_module.training_phase = "post_tune"
        elif pl_module.training_phase == "post_tune":
            self._post_tune_step += 1

            # Stop training after the maximum number of post-tuning steps is reached.
            if self._post_tune_step >= self.post_tune_steps:
                stop_train(trainer)

    def _update_reg_weight(self, pl_module: BaseLitModule) -> None:
        """Helper function to update and log the regularization parameter."""
        if self.reg_scheduler_update_every is not None and self._acip_step % self.reg_scheduler_update_every == 0:
            pl_module.objective.loss_weights["reg_objective"] *= self.reg_scheduler_update_factor
        pl_module.log("Training/reg_objective", pl_module.objective.loss_weights["reg_objective"], on_step=True)

    def _prepare_post_tune(self, pl_module: BaseLitModule) -> None:
        """
        Helper function to prepare the optimizer for post-tuning, i.e., tuning of the target parameters is disabled
        so that only `tuning_params` of `BaseOptimizerFactory` are optimized.
        """
        assert isinstance(pl_module.optimizer_factory, BaseOptimizerFactory)
        assert "target_params" in pl_module.optimizer_factory.params_grouped
        for param in pl_module.optimizer_factory.params_grouped["target_params"].values():
            param.requires_grad = False


class ScoreMapUpdater(pl.Callback, ABC):
    """
    Abstract base class of a Lightning callback that updates the score map of an `ACIPModel` over time.
    Every child class should make sure that a copy of `score_map` is stored in `self.results` after the fit.
    The keys should match those of `ACIPModel.score_map` whereas the values should be converted to flattened
    numpy arrays.
    """

    def __init__(self):
        super().__init__()
        self.results: dict[str, np.ndarray] = {}


class MaskScoreMapUpdater(ScoreMapUpdater):
    """
    Standard score map updater that implements the update step from the ACIP paper:
    After each training step, the score of all already pruned target parameters is decreased by 1 and all newly pruned
    target parameters are assigned score 0. The score of all non-pruned target parameters equals their value.
    """

    def __init__(self, stop_update_at_ratio: float | None = None, log_every_n_train_steps: int | None = 25):
        """
        Args:
            stop_update_at_ratio: Optional size ratio at which to stop updating the score map (earlier than ACIP). None or 0.0 mean
                that the score map is updated until the stopping criterion of ACIP is reached (default).
            log_every_n_train_steps: Frequency at which the score map is logged as heatmap to Weights & Biases.
                Use None to disable.
        """
        super().__init__()
        self.stop_update_at_ratio = stop_update_at_ratio
        self.log_every_n_train_steps = log_every_n_train_steps

        # Flag to indicate whether the score map has been initialized.
        self._score_map_initialized: bool = False

    def on_train_batch_end(
        self, trainer: pl.Trainer, pl_module: BaseLitModule, outputs: STEP_OUTPUT, batch: Any, batch_idx: int
    ) -> None:
        assert isinstance(pl_module.model, ACIPModel)

        # Don't update the score map if not in ACIP phase.
        if pl_module.training_phase != "acip":
            return

        # Don't update if `stop_update_at_ratio` was already reached.
        if (
            self.stop_update_at_ratio is not None
            and self.stop_update_at_ratio > 0.0
            and size_ratio_reached(pl_module, self.stop_update_at_ratio)
        ):
            return

        # Initialize the score map with the value of the target parameters.
        if not self._score_map_initialized:
            pl_module.model.score_map = pl_module.model.get_target_params()
            self._score_map_initialized = True
            logger.debug("Score map created.")

        # Perform the actual score map update.
        score_map = pl_module.model.score_map
        for p_name, param in pl_module.model.get_target_params().items():
            param_copy = param.clone().detach().float()
            score_map[p_name] = score_map[p_name].clone()
            score_map[p_name][param_copy > 0.0] = param_copy[param_copy > 0.0]
            new_zero_idx = (param_copy <= 0.0) & (score_map[p_name] >= 0.0)
            score_map[p_name][new_zero_idx] = 0.0
            score_map[p_name][score_map[p_name] <= 0.0] -= 1.0
        pl_module.model.score_map = score_map

        # Optionally log the score map as heatmap.
        if (
            wandb.run is not None
            and self.log_every_n_train_steps is not None
            and trainer.global_step % self.log_every_n_train_steps == 0
        ):
            score_map_plot = {}
            for p_name, score in pl_module.model.score_map.items():
                score_copy = score.clone().detach().cpu()
                # For better visualization, normalize the negative scores between 0 and 1.
                min_val = score_copy.min()
                if min_val < 0:
                    score_copy[score_copy < 0] = score_copy[score_copy < 0] / min_val.abs()
                score_map_plot[p_name] = score_copy
            fig = generate_params_plot(score_map_plot, zmin=None, zmax=None)
            wandb.log({"params_analysis/score_map": fig})

    def on_fit_end(self, trainer: pl.Trainer, pl_module: BaseLitModule) -> None:
        # Store the score map to `self.results` at the end of the fit.
        self.results["score_map"] = {  # type: ignore
            k: v.detach().flatten().float().cpu().numpy() for k, v in pl_module.model.score_map.items()
        }


class SVScoreMapUpdater(ScoreMapUpdater):
    """
    Singular-value-based score map "updater", which simply sets the score map equal to the singular values of the
    parametrized matrices at the beginning of the fit. So there are no updates.
    """

    def on_fit_start(self, trainer: pl.Trainer, pl_module: BaseLitModule) -> None:
        assert isinstance(pl_module.model, ACIPModel)

        score_map = {}
        for m_name, module in pl_module.model.parametrized_modules.items():
            # This only works for SVD parametrizations.
            assert isinstance(module.parametrization, SVDLinearParametrization)
            for p_name in module.parametrization.get_target_params().keys():
                score_map[f"{m_name}.parametrization.{p_name}"] = module.parametrization.get_buffer("S").float()
        pl_module.model.score_map = score_map
        logger.debug("Score map created.")

        # Store the score map in `self.results`.
        self.results["score_map"] = {  # type: ignore
            k: v.detach().flatten().float().cpu().numpy() for k, v in pl_module.model.score_map.items()
        }

        # Log the score map as heatmap.
        if wandb.run is not None:
            score_map_plot = {}
            for p_name, score in pl_module.model.score_map.items():
                score_copy = score.clone().detach().cpu()
                score_map_plot[p_name] = torch.log(score_copy + 1.0)
            fig = generate_params_plot(score_map_plot, zmin=None, zmax=None)
            wandb.log({"params_analysis/score_map": fig})
