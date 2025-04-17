from typing import Any

import torch
import wandb
from lightning import pytorch as pl
from lightning_utilities.core.rank_zero import rank_zero_only

from acip.core.acip_model import ACIPModel
from acip.eval.evaluator import ModelEvaluator
from acip.training.monitoring import logger
from acip.training.pl_module import BaseLitModule
from acip.training.utils import create_eval_dataframe, generate_params_plot


class ModelBenchmarker(pl.Callback):
    """
    Lightning callback that benchmarks an `ACIPModel` at the beginning and end of fit.
    Compared to `ModelMonitor` the evaluation can be more extensive.
    All results are stored in `self.results` and can be saved after training.

    Notes: We are using on_fit_start/end callback hooks instead of on_train_start/end because in some cases,
        the trainer does not perform training, for example, when a model is loaded from disk and just evaluated.
    """

    def __init__(
        self,
        evaluator: ModelEvaluator,
        test_ratios: list[float] | None = None,
        measure_ratio_full: bool = False,
    ):
        """
        Args:
            evaluator: `ModelEvaluator` instance to use for benchmarking.
            test_ratios: If a list of floats between 0 and 1 is provided, the model will be pruned to each of these
                compression ratios (via `ACIPModel.prune_model_by_score`) and evaluated.
                If `None`, the model will be evaluated at the end of training as it is.
            measure_ratio_full: If `True`, all parameters of the model are counted when pruning to a target ratio
                is performed, if `False` only the parameters of the parametrized modules are counted (default).
                See `full` flag in `ACIPModel.prune_model_by_score`.
        """
        self.evaluator = evaluator
        self.test_ratios = test_ratios
        self.measure_ratio_full = measure_ratio_full

        # Stores evaluation results. The key indicates the point of evaluation and
        # the value contains the actual results as a dict (output of `self.evaluator(...)`).
        self.results: dict[str, dict[str, Any]] = {}

    @rank_zero_only
    def on_fit_start(self, trainer: pl.Trainer, pl_module: BaseLitModule) -> None:
        logger.info("Benchmarking model at train start.")
        self.results["at_start"] = self.evaluator(model=pl_module.model)
        if wandb.run is not None:
            wandb.log({"eval/model": wandb.Table(dataframe=create_eval_dataframe(self.results))})

    @rank_zero_only
    def on_fit_end(self, trainer: pl.Trainer, pl_module: BaseLitModule) -> None:
        # Lightning's teardown may move the model to CPU. Move it back to GPU if possible.
        was_on_cpu = pl_module.device.type == "cpu"
        if was_on_cpu and torch.cuda.is_available():
            pl_module.cuda()

        if self.test_ratios is not None:
            # This only works for `ACIPModel`s
            assert isinstance(pl_module.model, ACIPModel)
            # Benchmark model at each compression ratio
            for ratio in self.test_ratios:
                logger.info(f"Benchmarking model at compression ratio {ratio}.")
                pl_module.model.prune_model_by_score(compression_ratio=ratio, full=self.measure_ratio_full)
                self.results[f"at_end_ratio{ratio}"] = self.evaluator(model=pl_module.model)
                # Add target compression ratio to results for better identification
                self.results[f"at_end_ratio{ratio}"]["target_compression_ratio"] = ratio

                if wandb.run is not None:
                    fig = generate_params_plot(pl_module.model.get_target_params(), zmin=None, zmax=None)
                    wandb.log({f"params_analysis/target_params_ratio{ratio}": fig})
        else:
            logger.info("Benchmarking model at train end.")
            self.results["at_end"] = self.evaluator(model=pl_module.model)

        if wandb.run is not None:
            wandb.log({"eval/model": wandb.Table(dataframe=create_eval_dataframe(self.results))})

        # Move model back to CPU if it was before evaluation.
        if was_on_cpu:
            pl_module.cpu()
