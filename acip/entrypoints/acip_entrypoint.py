import contextlib
import os
import time
from logging import getLogger
from typing import Any

import hydra
import numpy as np
import torch
import wandb
from accelerate import init_on_device
from lightning import pytorch as pl
from lightning.pytorch.loggers import WandbLogger
from omegaconf import DictConfig, OmegaConf

from acip.data.dataset import DatasetFactory
from acip.model.model_factory import ACIPModelFactory
from acip.training.acip import ScoreMapUpdater
from acip.training.benchmarking import ModelBenchmarker
from acip.training.monitoring import ACIPEfficiencyMonitor, ModelMonitor
from acip.training.pl_module import BaseDataModule, BaseLitModule
from acip.utils.utils import clear_cache
from scaffold.conf.scaffold.entrypoint import EntrypointConf
from scaffold.ctx_manager import WandBContext
from scaffold.entrypoints import Entrypoint
from scaffold.hydra.config_helpers import structured_config

# Custom list merge resolver (required to merge cft.run.tags_default and cft.run.tags_custom).
OmegaConf.register_new_resolver("merge", lambda x, y: x + y)

logger = getLogger(__name__)


@structured_config(group="acip")
class ACIPEntrypointConf(EntrypointConf):
    """
    Structured config for `ACIPEntrypoint`. Note that we only expose the top-level sub-configs here.
    Each sub-config is a (nested) `DictConfig` if not None. See "config" directory for more details.

    See Also:
        `ACIPEntrypoint`
    """

    # Not required because we don't use scaffold hydra logging.
    verbose = False

    # Expected config subpackages.
    paths: Any | None = None
    run: Any | None = None
    wandb: Any | None = None
    data: Any | None = None
    eval: Any | None = None
    model: Any | None = None
    training: Any | None = None
    acip: Any | None = None


class ACIPEntrypoint(Entrypoint[ACIPEntrypointConf]):
    """
    An mxm-scaffold `Entrypoint` implementation for ACIP. See `run` for the executed code.
    The entrypoint config is of type `ACIPEntrypointConf` and can be accessed via `self.config`.

    See Also:
        `ACIPEntrypointConf`
    """

    def run(self) -> Any:
        """Run the ACIP entrypoint."""

        # Add the actual run url to the wandb config so that is stored in the run results later
        # and is therefore easier to identify.
        if wandb.run is not None:
            OmegaConf.set_struct(self.config.wandb, False)
            self.config.wandb["url"] = wandb.run.get_url()
            OmegaConf.set_struct(self.config.wandb, True)

        # --- Set Lightning seed for reproducibility ---
        pl.seed_everything(self.config.training.seed)

        # --- Create dataset factories and Lightning data module ---
        train_dataset_factory: DatasetFactory = hydra.utils.instantiate(
            self.config.data.train_dataset_factory, _convert_="object"
        )
        logger.info("Training data factory created.")
        val_dataset_factory: DatasetFactory | None = None
        if self.config.data.val_dataset_factory is not None:
            val_dataset_factory = hydra.utils.instantiate(self.config.data.val_dataset_factory, _convert_="object")
            logger.info("Validation data factory created.")
        test_dataset_factory: DatasetFactory | None = None
        if self.config.data.test_dataset_factory is not None:
            test_dataset_factory = hydra.utils.instantiate(self.config.data.test_dataset_factory, _convert_="object")
            logger.info("Test data factory created.")

        data_module = BaseDataModule(
            train_dataset_factory=train_dataset_factory,
            val_dataset_factory=val_dataset_factory,
            test_dataset_factory=test_dataset_factory,
            **self.config.training.data_module,
        )
        logger.info("Data module created.")

        # When using "high" precision, float32 multiplications may use a bfloat16-based algorithm that is more
        # complicated than simply truncating to some smaller number mantissa bits
        # (e.g. 10 for TensorFloat32, 8 for bfloat16).
        torch.set_float32_matmul_precision(self.config.training.torch_float32_matmul_precision)

        # --- Create model factory ---
        model_factory: ACIPModelFactory = hydra.utils.instantiate(self.config.model.model_factory, _convert_="object")
        logger.info("Model factory created.")

        # If cache_init_model is True, the initial model will be created right away and saved to acip.save.path.
        # Note that this is the same path that will be used for the final model, i.e., the cached model will be
        # overwritten with the final model. However, this caching step can be useful for debugging purposes because
        # the creation of a `ParametrizedModel` can take a long time. If cache_init_model is False, the model will
        # be created from scratch by the the `configure_model` hook of the `BaseLitModule` (otherwise just loaded).
        # Also note that if a model already exists at acip.save.path, it will not be overwritten. Here it does not
        # matter if this was a final model or the initial model.
        if (
            "cache_init_model" in self.config.acip
            and self.config.acip.cache_init_model
            and "model" in self.config.run.save
        ):
            logger.info("Creating and caching initial parametrized model...")
            cache_init_model_path = self.config.acip.save.path
            if not os.path.exists(cache_init_model_path):
                with init_on_device(device=torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")):
                    model = model_factory()
                # Don't use any filters because we need to store the full model.
                model.save_pretrained(
                    cache_init_model_path, include_filter=None, exclude_filter=None, **self.config.acip.save.kwargs
                )
                del model
                clear_cache(gc=True)
                logger.info(f"Parametrized model saved to {cache_init_model_path}.")
            else:
                logger.info(f"A parametrized model already exists at {cache_init_model_path}.")
            # We reset the model config to make sure that the `BaseLitModule` really loads the model from the cache
            # and not from the config.
            model_factory.pretrained_model_name_or_path = cache_init_model_path
            model_factory.pretrained_model_config = None

        # --- Create BaseLitModule including Objective and Optimizer & Scheduler Factories ---
        pl_module = BaseLitModule(
            objective=hydra.utils.instantiate(self.config.training.objective, _convert_="object"),
            model_factory=model_factory,
            optimizer_factory=hydra.utils.instantiate(self.config.training.optimizer_factory, _convert_="object"),
            scheduler_factory=hydra.utils.instantiate(self.config.training.scheduler_factory, _convert_="object"),
        )
        logger.info("BaseLitModule created.")

        # --- Create Lightning callbacks ---
        pl_callbacks: dict[str, pl.Callback] = hydra.utils.instantiate(
            self.config.training.callbacks, _convert_="object"
        )
        logger.info("Lightning callbacks created.")

        # --- Create Lightning trainer ---
        pl_trainer = pl.Trainer(
            strategy=pl_module.create_pl_strategy(),
            callbacks=[callback for callback in pl_callbacks.values()],
            # The WandbLogger should connect to the same run as the one created by the ACIPEntrypoint.
            logger=WandbLogger() if self.config.wandb is not None else False,
            **self.config.training.trainer,
        )
        logger.info("Lightning trainer created.")

        # --- Perform training ---
        logger.info("Starting training.")
        time_start = time.time()
        pl_trainer.fit(model=pl_module, datamodule=data_module)
        time_stop = time.time()
        logger.info(f"Training completed after {round((time_stop - time_start) / 60, 2)}mins.")

        # --- Save run results ---
        if "results" in self.config.run.save:
            # --- Save target parameters & score map ---
            params_analysis = {}
            # Target parameters are stored in the `model_monitor` callback (`ModelMonitor`).
            model_monitor = pl_callbacks.get("model_monitor", None)
            if model_monitor is not None and isinstance(model_monitor, ModelMonitor):
                params_analysis["target_params"] = model_monitor.results.get("target_params", {})
            # Score map is stored in the `score_map_updater` callback (`ScoreMapUpdater`).
            score_map_updater = pl_callbacks.get("score_map_updater", None)
            if score_map_updater is not None and isinstance(score_map_updater, ScoreMapUpdater):
                params_analysis["score_map"] = score_map_updater.results.get("score_map", {})
            # Save into a .npz file
            if len(params_analysis) > 0:
                np.savez_compressed(os.path.join(self.config.run.path, "params_analysis.npz"), **params_analysis)
                logger.info(f"Params analysis saved to {self.config.run.path}.")

            # --- Save config & benchmark results ---
            results = {
                "run_cfg": self.config,
                "train_runtime": time_stop - time_start,  # Include fit runtime
            }
            # Benchmark results are stored in the `model_benchmarker` callback (`ModelBenchmarker`).
            model_benchmarker = pl_callbacks.get("model_benchmarker", None)
            if model_benchmarker is not None and isinstance(model_benchmarker, ModelBenchmarker):
                results["benchmark"] = OmegaConf.create(model_benchmarker.results)
            # Benchmark results stored in BaseLitModule.
            results["lit_module"] = pl_module.results
            # ACIP efficiency analysis results if available.
            acip_efficiency_monitor = pl_callbacks.get("acip_efficiency_monitor", None)
            if acip_efficiency_monitor is not None and isinstance(acip_efficiency_monitor, ACIPEfficiencyMonitor):
                results["acip_efficiency"] = OmegaConf.create(acip_efficiency_monitor.results)
            # Save into a .yaml file
            OmegaConf.save(results, f=os.path.join(self.config.run.path, "results.yaml"))
            logger.info(f"Results saved to {self.config.run.path}.")

        # --- Save ACIP model ---
        if "model" in self.config.run.save:
            pl_module.model.save_pretrained(
                self.config.acip.save.path,
                include_filter=self.config.acip.save.include_filter,
                exclude_filter=self.config.acip.save.exclude_filter,
                **self.config.acip.save.kwargs,
            )
            logger.info(f"ACIP model saved to {self.config.acip.save.path}.")

        clear_cache(gc=True)


def run_acip_entrypoint(cfg: DictConfig | ACIPEntrypointConf) -> None:
    """
    Main function to create and execute an ACIPEntrypoint.
    In the actual entrypoint scripts, this function should be wrapped accordingly by a hydra config setting.
    """
    # Resolve config to make print-outs easily readable.
    OmegaConf.resolve(cfg)

    # Print config
    if cfg.run.print_cfg:
        logger.info(f"Full hydra config:\n{OmegaConf.to_yaml(cfg)}")
    # Stop here if dry run
    if cfg.run.dry_run:
        return
    # Save config to run directory
    if "config" in cfg.run.save:
        cfg_path = os.path.join(cfg.run.path, "hydra", "config.yaml")
        OmegaConf.save(cfg, f=cfg_path, resolve=True)
        logger.info(f"Hydra config saved to {cfg_path}.")

    # Build WandB context if applicable.
    # We cannot define this context directly in ACIPEntrypoint's contexts attribute because
    # we would like to log the run config as well, which is only rendered at runtime.
    run_ctx = contextlib.nullcontext()
    if cfg.wandb is not None and os.environ.get("LOCAL_RANK", "0") == "0":
        # The wandb run dir is not created automatically.
        if "dir" in cfg.wandb:
            os.makedirs(cfg.wandb.dir, exist_ok=True)
        # Generate a unique but human-readable run ID if not provided
        if "name" in cfg.wandb and "run_id" not in cfg.wandb:
            wandb_run_id = {"run_id": cfg.wandb.name + "__" + wandb.util.generate_id()}
        else:
            wandb_run_id = {}
        # Pass the run config to the WandB context
        run_ctx = WandBContext(**cfg.wandb, run_config=cfg, **wandb_run_id)

    entrypoint = ACIPEntrypoint(config=cfg, contexts=[run_ctx])
    entrypoint()
