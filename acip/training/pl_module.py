import time
from abc import ABC
from logging import getLogger
from typing import Any, Callable, Type

import lightning.pytorch as pl
import torch
from lightning.pytorch.strategies import ParallelStrategy
from lightning.pytorch.utilities.types import STEP_OUTPUT
from torch.utils.data import DataLoader
from transformers import PreTrainedModel

from acip.data.dataset import DatasetFactory
from acip.model.model_factory import PretrainedModelFactory
from acip.training.objective import Objective
from acip.training.optimizer import OptimizerFactory
from acip.utils.utils import clear_cache

logger = getLogger(__name__)


class BaseLitModule(pl.LightningModule, ABC):
    """
    Base LightningModule for training a HuggingFace `PreTrainedModel`.
    It is intentionally kept generic so that it does not depend on any ACIP functionality or NLP-specific code.

    The model and optimizer are initialized lazily in `configure_model` and `configure_optimizers` via
    respective factories.
    This makes the training code compatible with more advanced training strategies like DDP or FSDP.

    The forward step passes the model and current batch to an instance of `Objective`, which is responsible for
    computing the loss and accompanying metrics.
    """

    def __init__(
        self,
        objective: Objective,
        model_factory: PretrainedModelFactory,
        optimizer_factory: OptimizerFactory,
        scheduler_factory: Callable[..., torch.optim.lr_scheduler] | None = None,
    ):
        """
        Args:
            objective: The training objective to be used.
            model_factory: Factory to create the model with the `configure_model` hook.
            optimizer_factory: Factory to create the optimizer with the `configure_optimizers` hook.
            scheduler_factory: Optional callable that creates a learning rate scheduler with
                the `configure_optimizers` hook.
        """
        super().__init__()
        # important to ignore the factories because they would be called unintentionally
        self.save_hyperparameters(
            logger=False,
            ignore=["objective", "model_factory", "optimizer_factory", "scheduler_factory"],
        )

        self.objective = objective
        self.model_factory = model_factory
        self.optimizer_factory = optimizer_factory
        self.scheduler_factory = scheduler_factory

        self._training_phase = None

        self.model: PreTrainedModel | None = None

        # Benchmarking metrics can be stored here and will be saved by ACIPEntrypoint.
        self.results: dict[str, Any] = {}

    @property
    def training_phase(self) -> str | None:
        """
        Marker of the current training phase. Should be managed by Lightning callbacks.
        Can be ignored if there is only a single training phase.
        """
        return self._training_phase

    @training_phase.setter
    def training_phase(self, value: str | None) -> None:
        self._training_phase = value
        logger.info(f"Entering training phase '{value}' at step {self.trainer.global_step}.")
        logger.debug("Tunable parameters during compression:")
        for p_name, param in self.model.named_parameters():
            if param.requires_grad:
                logger.debug(f"Optimizer - param: {p_name}")

    def configure_model(self) -> None:
        if self.model is None:
            start_time = time.time()
            self.model = self.model_factory()
            self.results["model_creation_time"] = time.time() - start_time
            logger.info(f"Model created by factory in {self.results['model_creation_time'] / 60:.2f} min.")
            logger.debug(f"Model structure:\n{self.model}")
            clear_cache(gc=True)

    def create_pl_strategy(self) -> str | ParallelStrategy:
        """
        Create a Lightning `ParallelStrategy` used by the Lightning Trainer.
        This may also depend on the underlying base model, e.g., taking into account that specific layers of the model
        should not be sharded. So far, only `SingleDeviceStrategy` was tested.
        """
        return "auto"

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, Any]:
        return self.objective(batch, self.model)

    def _step(self, batch: dict[str, torch.Tensor], prefix: str, log_kwargs: dict[str, Any]) -> STEP_OUTPUT:
        """Helper function for training, validation and testing steps. Performs the forward pass and logs metrics."""
        output = self.forward(batch)
        loss = output["loss"]
        # Extract additional metrics using the "info_" convention from `Objective`
        info = {k: v for k, v in output.items() if k.startswith("info_")}

        self.log("Training/" + prefix + "_loss", loss.item(), **log_kwargs)
        for k, v in info.items():
            self.log("Training/" + prefix + "_" + k, v, **log_kwargs)
        return {"loss": loss}

    def training_step(self, batch: dict[str, torch.Tensor], batch_idx) -> STEP_OUTPUT:
        return self._step(batch, "train", {"prog_bar": True, "on_step": True, "sync_dist": True})

    def validation_step(self, batch: dict[str, torch.Tensor], batch_idx) -> STEP_OUTPUT:
        return self._step(batch, "val", {"on_epoch": True, "sync_dist": True})

    def test_step(self, batch: dict[str, torch.Tensor], batch_idx) -> STEP_OUTPUT:
        return self._step(batch, "test", {"prog_bar": True, "on_step": True, "sync_dist": True})

    def configure_optimizers(self) -> dict[str, Type[torch.optim.Optimizer] | torch.optim.lr_scheduler]:
        optimizer = self.optimizer_factory(self.model)
        if self.scheduler_factory is not None:
            scheduler = self.scheduler_factory(optimizer)
            return {"optimizer": optimizer, "lr_scheduler": scheduler}
        else:
            return {"optimizer": optimizer}


class BaseDataModule(pl.LightningDataModule):
    """
    Basic `LightningDataModules` to be used in conjunction with `BaseLitModule`.
    It uses the concept of `DatasetFactory` to create train, val and test datasets, and generates corresponding
    PyTorch `DataLoader`s.
    """

    def __init__(
        self,
        train_dataset_factory: DatasetFactory | None = None,
        val_dataset_factory: DatasetFactory | None = None,
        test_dataset_factory: DatasetFactory | None = None,
        train_batch_size: int = 4,
        val_batch_size: int | None = None,
        test_batch_size: int | None = None,
        num_workers: int = 0,
    ):
        """
        Args:
            train_dataset_factory: Training dataset factory.
            val_dataset_factory: Validation dataset factory.
            test_dataset_factory: Test dataset factory.
            train_batch_size: Training batch size.
            val_batch_size: Validation batch size. If None, defaults to `train_batch_size`.
            test_batch_size: Test batch size. If None, defaults to `val_batch_size`.
            num_workers: Number of workers used for data loaders.
        """
        super().__init__()

        self.train_dataset_factory = train_dataset_factory
        self.val_dataset_factory = val_dataset_factory
        self.test_dataset_factory = test_dataset_factory
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size if val_batch_size is not None else train_batch_size
        self.test_batch_size = test_batch_size if test_batch_size is not None else val_batch_size
        self.num_workers = num_workers

        self.datasets = {}

    def prepare_data(self):
        # download dataset metadata
        if self.train_dataset_factory is not None:
            self.train_dataset_factory()
        if self.val_dataset_factory is not None:
            self.val_dataset_factory()
        if self.test_dataset_factory is not None:
            self.test_dataset_factory()

    def setup(self, stage: str = None):
        # Create datasets and prepare them for data loading by processing
        train_dataset = self.train_dataset_factory() if self.train_dataset_factory is not None else None
        if train_dataset is not None:
            self.datasets["train"] = self.train_dataset_factory.process_for_model_input(train_dataset)
        else:
            self.datasets["train"] = None

        val_dataset = self.val_dataset_factory() if self.val_dataset_factory is not None else None
        if val_dataset is not None:
            self.datasets["val"] = self.val_dataset_factory.process_for_model_input(val_dataset)
        else:
            self.datasets["val"] = None

        test_dataset = self.test_dataset_factory() if self.test_dataset_factory is not None else None
        if test_dataset is not None:
            self.datasets["test"] = self.test_dataset_factory.process_for_model_input(test_dataset)
        else:
            self.datasets["test"] = None

    def train_dataloader(self) -> DataLoader:
        if self.datasets["train"] is None:
            raise ValueError("No training dataset found.")
        # Shuffling is already handled by the dataset factory
        return DataLoader(
            self.datasets["train"], batch_size=self.train_batch_size, shuffle=False, num_workers=self.num_workers
        )

    def val_dataloader(self) -> DataLoader:
        if self.datasets["val"] is None:
            raise ValueError("No validation dataset found.")
        # Shuffling is already handled by the dataset factory
        return DataLoader(
            self.datasets["val"], batch_size=self.val_batch_size, shuffle=False, num_workers=self.num_workers
        )

    def test_dataloader(self) -> DataLoader:
        if self.datasets["test"] is None:
            raise ValueError("No test dataset found.")
        # Shuffling is already handled by the dataset factory
        return DataLoader(
            self.datasets["test"], batch_size=self.test_batch_size, shuffle=False, num_workers=self.num_workers
        )
