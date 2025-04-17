import time
from logging import getLogger
from typing import Any

import numpy as np
import torch
import wandb
from lightning import pytorch as pl
from lightning.pytorch.utilities import grad_norm
from lightning.pytorch.utilities.types import STEP_OUTPUT
from lightning_utilities.core.rank_zero import rank_zero_only
from tqdm import tqdm

from acip.core.acip_model import ACIPModel
from acip.core.parametrized_model import ParametrizedModel
from acip.eval.evaluator import ModelEvaluator
from acip.training.pl_module import BaseLitModule
from acip.training.utils import generate_params_plot

logger = getLogger(__name__)


class GradientMonitor(pl.Callback):
    """
    Lightning callback that logs the 2-norm of the gradients for each model parameter.
    """

    def __init__(self, log_every_n_train_steps: int | None = 25):
        """
        Args:
            log_every_n_train_steps: Log frequency, use None to disable.
        """
        super().__init__()
        self.log_every_n_train_steps = log_every_n_train_steps

    @rank_zero_only
    def on_after_backward(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if self.log_every_n_train_steps is None or trainer.global_step % self.log_every_n_train_steps != 0:
            return

        # Compute the 2-norm for each layer
        # If using mixed precision, the gradients are already unscaled here
        norms = grad_norm(pl_module, norm_type=2.0)
        pl_module.log_dict(norms)


class ModelMonitor(pl.Callback):
    """
    Lightning callback that monitors a `ParametrizedModel` over training and logs the results.
    """

    def __init__(
        self,
        train_evaluator: ModelEvaluator | None = None,
        val_evaluator: ModelEvaluator | None = None,
        log_target_params: bool = True,
        zrange_target_params: tuple[float | None, float | None] = (0.0, None),
        log_every_n_train_steps: int | None = 25,
    ):
        """
        Args:
            train_evaluator: `ModelEvaluator` that evaluates the model every `log_every_n_train_steps`.
            val_evaluator: `ModelEvaluator` that evaluates the model at the beginning of each validation epoch.
            log_target_params: If True, the target parameters of the `ParametrizedModel` are saved in `self.results`
                every `log_every_n_train_steps`. Moreover, the target parameters are logged as heatmaps to
                Weights & Biases.
            zrange_target_params: Heatmap z-range when logging target parameters. The first value is the minimum,
                the second value is the maximum, see also `generate_params_plot`.
            log_every_n_train_steps: Log frequency, use None to disable.
        """
        super().__init__()
        self.train_evaluator = train_evaluator
        self.val_evaluator = val_evaluator
        self.log_target_params = log_target_params
        self.zrange_target_params = zrange_target_params
        self.log_every_n_train_steps = log_every_n_train_steps

        # Stores the target parameters
        self.results = {}

    @rank_zero_only
    def on_validation_epoch_start(self, trainer: pl.Trainer, pl_module: BaseLitModule) -> None:
        logger.debug("Validation epoch started.")
        if self.val_evaluator is not None:
            result = self.val_evaluator(model=pl_module.model)
            for k, v in result.items():
                pl_module.log("monitoring/" + k, v, on_epoch=True)

    @rank_zero_only
    def on_train_batch_end(
        self, trainer: pl.Trainer, pl_module: BaseLitModule, outputs: STEP_OUTPUT, batch: Any, batch_idx: int
    ) -> None:
        if self.log_every_n_train_steps is None or trainer.global_step % self.log_every_n_train_steps != 0:
            return

        if self.train_evaluator is not None:
            results = self.train_evaluator(model=pl_module.model)
            for k, result in results.items():
                pl_module.log("monitoring/" + k, result, on_step=True)

        if self.log_target_params:
            # This only works for `ParametrizedModel`
            assert isinstance(pl_module.model, ParametrizedModel)
            # Store a copy of the current target parameters in `self.results`
            target_params = pl_module.model.get_target_params()
            target_params_np = {k: v.detach().flatten().float().cpu().numpy() for k, v in target_params.items()}
            if "target_params" not in self.results:
                self.results["target_params"] = {}
            self.results["target_params"][trainer.global_step] = target_params_np

            if wandb.run is not None:
                fig = generate_params_plot(
                    target_params,
                    zmin=self.zrange_target_params[0],
                    zmax=self.zrange_target_params[1],
                )
                wandb.log({"params_analysis/target_params": fig})


class ACIPEfficiencyMonitor(pl.Callback):
    """
    Lightning callback that monitors the ACIP tuning efficiency.
    All results are stored in `self.results` and can be saved after training.
    Specifically, the returned dictionary has the following keys:
     - train_time: Total training time in seconds.
     - train_step_time: Average train step time in seconds.
     - pruning_time: Average pruning time in seconds (via `model.prune_model_by_score`).
     - compress_time: Time to actually compress the model in seconds (via `model.compress`).
     - peak_memory_allocated: The peak memory allocated by the model in GB.
     - peak_memory_reserved: The peak memory reserved by the model in GB.
    """

    def __init__(self, min_compression_ratio: float = 0.4):
        """
        Args:
            min_compression_ratio: Minimum compression ratio that pruning will be tested with.
        """
        super().__init__()
        self.min_compression_ratio = min_compression_ratio

        # Stores the monitored results
        self.results: dict[str, Any] = {}

        # Helpers to track average train step time
        self._train_step_count: int = 0
        self._train_step_time: float = 0
        self.results["train_step_time"] = 0
        # Helpers to store GPU memory peak consumption
        self._peak_memory_allocated: float = 0
        self._peak_memory_reserved: float = 0

    def on_train_start(self, trainer: pl.Trainer, pl_module: BaseLitModule) -> None:
        # Start tracking total train time
        self.results["train_time"] = time.time()

        torch.cuda.reset_peak_memory_stats(pl_module.model.device)

    def on_train_end(self, trainer: pl.Trainer, pl_module: BaseLitModule) -> None:
        # Compute average train step time and total train time
        self.results["train_step_time"] = self.results["train_step_time"] / self._train_step_count
        self.results["train_time"] = time.time() - self.results["train_time"]

        assert isinstance(pl_module.model, ACIPModel)

        # Randomly prune the model to estimate the average pruning time
        self.results["pruning_time"] = time.time()
        num_tests = 20
        for _ in tqdm(range(num_tests), desc="Measuring pruning time"):
            compression_ratio = np.random.uniform(self.min_compression_ratio, 1.0)
            pl_module.model.prune_model_by_score(compression_ratio=compression_ratio)
        self.results["pruning_time"] = (time.time() - self.results["pruning_time"]) / num_tests

        # Measure time of actual compression
        pl_module.model.prune_model_by_score(compression_ratio=self.min_compression_ratio)
        self.results["compress_time"] = time.time()
        pl_module.model.compress()
        self.results["compress_time"] = time.time() - self.results["compress_time"]

        self.results["peak_memory_allocated"] = self._peak_memory_allocated / (1024**3)
        self.results["peak_memory_reserved"] = self._peak_memory_reserved / (1024**3)

        logger.info(f"ACIP efficiency results: {self.results}")

    def on_train_batch_start(self, trainer: pl.Trainer, pl_module: BaseLitModule, batch: Any, batch_idx: int) -> None:
        # Start tracking time of the current train step
        self._train_step_time = time.time()

    def on_train_batch_end(
        self, trainer: pl.Trainer, pl_module: BaseLitModule, outputs: STEP_OUTPUT, batch: Any, batch_idx: int
    ) -> None:
        # Add the time of the current train step
        self.results["train_step_time"] += time.time() - self._train_step_time
        self._train_step_count += 1

        # Track GPU memory peak consumption here
        if torch.cuda.is_available():
            self._peak_memory_allocated = max(
                self._peak_memory_allocated, torch.cuda.max_memory_allocated(pl_module.model.device)
            )
            self._peak_memory_reserved = max(
                self._peak_memory_reserved, torch.cuda.max_memory_reserved(pl_module.model.device)
            )
