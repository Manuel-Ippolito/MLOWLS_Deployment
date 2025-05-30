from typing import Any, Dict, Optional

import torch
from rich.console import Console

from interfaces.model.logger import Logger
from interfaces.model.model_saver import ModelSaver
from interfaces.model.training_strategy import TrainingStrategy
from ml_owls.model.metrics.aggregator import MetricAggregator
from ml_owls.model.training.epoch_trainer import EpochTrainer


class Trainer:
    """Main trainer orchestrator following Dependency Inversion Principle."""

    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        criterion: torch.nn.Module,
        dataloaders: Dict[str, torch.utils.data.DataLoader],
        device: torch.device,
        logger: Logger,
        model_saver: ModelSaver,
        strategy: TrainingStrategy,
        metric_aggregator: MetricAggregator,
        config: Any,
    ) -> None:
        """Initialize trainer with all dependencies.

        Args:
            model: Neural network model
            optimizer: Optimizer instance
            criterion: Loss function
            dataloaders: Dictionary of train/val dataloaders
            device: Device to run computations on
            logger: Logger implementation
            model_saver: Model saver implementation
            strategy: Training strategy implementation
            metric_aggregator: Metrics aggregator
            config: Configuration object
        """
        self.model = model.to(device)
        self.optimizer = optimizer
        self.criterion = criterion
        self.dataloaders = dataloaders
        self.device = device
        self.logger = logger
        self.model_saver = model_saver
        self.config = config

        console = Console()
        self.epoch_trainer = EpochTrainer(device, strategy, metric_aggregator, console)
        self.console = console

    def fit(self, epochs: int) -> None:
        """Execute complete training loop.

        Args:
            epochs: Number of epochs to train
        """
        best_val_metric = 0.0
        patience_counter = 0
        max_patience = getattr(self.config, "early_stopping_patience", 10)

        self.console.log(f"[bold blue]Starting training for {epochs} epochs...")
        self.console.log(f"[blue]Strategy: {self.epoch_trainer.strategy.name}")
        self.console.log(
            f"[blue]Metrics: {', '.join(self.epoch_trainer.metric_aggregator.get_metric_names())}"
        )

        for epoch in range(1, epochs + 1):
            # Train epoch
            train_metrics = self.epoch_trainer.train_epoch(
                self.model, self.dataloaders["train"], self.optimizer, self.criterion, epoch
            )

            # Validate epoch
            val_metrics = self.epoch_trainer.validate_epoch(
                self.model, self.dataloaders.get("val"), self.criterion, epoch
            )

            # Log all metrics
            self._log_metrics("train", train_metrics, epoch)
            if val_metrics:
                self._log_metrics("val", val_metrics, epoch)

                # Model saving and early stopping logic
                current_val_metric = val_metrics.get("auc", val_metrics.get("accuracy", 0))

                if current_val_metric > best_val_metric:
                    best_val_metric = current_val_metric
                    patience_counter = 0

                    # Save best model
                    best_model_path = f"{self.config.output_dir}/best_model.pth"
                    metadata = {
                        "epoch": epoch,
                        "val_metrics": val_metrics,
                        "train_metrics": train_metrics,
                        "strategy": self.epoch_trainer.strategy.name,
                    }

                    self.model_saver.save_model(self.model, best_model_path, metadata)
                    self.console.log(
                        f"[bold green]💾 New best model saved! "
                        f"Val metric: {current_val_metric:.4f}"
                    )

                    # Log as artifact
                    self.logger.log_artifact(best_model_path, artifact_path="models")
                else:
                    patience_counter += 1

                # Early stopping check
                if patience_counter >= max_patience:
                    self.console.log(
                        f"[bold yellow]⏹️ Early stopping triggered after {patience_counter} epochs "
                        f"without improvement"
                    )
                    break

            # Log epoch summary
            self._log_epoch_summary(epoch, train_metrics, val_metrics)

        # Save final model
        final_model_path = f"{self.config.output_dir}/final_model.pth"
        final_metadata = {
            "epoch": epoch,
            "final_train_metrics": train_metrics,
            "final_val_metrics": val_metrics,
            "best_val_metric": best_val_metric,
            "strategy": self.epoch_trainer.strategy.name,
        }

        self.model_saver.save_model(self.model, final_model_path, final_metadata)
        self.logger.log_artifact(final_model_path, artifact_path="models")

        self.console.log(
            f"[bold green]🎉 Training completed! Best val metric: {best_val_metric:.4f}"
        )

    def _log_metrics(self, prefix: str, metrics: Dict[str, float], epoch: int) -> None:
        """Log metrics with prefix to logger.

        Args:
            prefix: Metric prefix (train/val)
            metrics: Dictionary of metrics
            epoch: Current epoch number
        """
        for metric_name, value in metrics.items():
            self.logger.log_metric(f"{prefix}_{metric_name}", value, step=epoch)

    def _log_epoch_summary(
        self, epoch: int, train_metrics: Dict[str, float], val_metrics: Optional[Dict[str, float]]
    ) -> None:
        """Log epoch summary to logger.

        Args:
            epoch: Current epoch number
            train_metrics: Training metrics
            val_metrics: Validation metrics (optional)
        """
        summary = {
            "epoch": epoch,
            "train_loss": train_metrics.get("loss", 0),
            "train_accuracy": train_metrics.get("accuracy", 0),
        }

        if val_metrics:
            summary.update(
                {
                    "val_loss": val_metrics.get("loss", 0),
                    "val_accuracy": val_metrics.get("accuracy", 0),
                    "val_auc": val_metrics.get("auc", 0),
                }
            )

        # Log summary as batch metrics
        self.logger.log_batch_metrics(summary, step=epoch)

    def evaluate(self, dataloader: torch.utils.data.DataLoader) -> Dict[str, float]:
        """Evaluate model on given dataloader.

        Args:
            dataloader: DataLoader to evaluate on

        Returns:
            Dictionary of calculated metrics
        """
        return (
            self.epoch_trainer.validate_epoch(self.model, dataloader, self.criterion, epoch=0) or {}
        )

    def predict(self, dataloader: torch.utils.data.DataLoader) -> torch.Tensor:
        """Generate predictions for given dataloader.

        Args:
            dataloader: DataLoader to predict on

        Returns:
            Tensor of predictions
        """
        self.model.eval()
        predictions = []

        with torch.no_grad():
            for batch in dataloader:
                specs, _ = batch
                specs = specs.to(self.device)
                preds = self.model(specs)
                probs = torch.nn.functional.softmax(preds, dim=1)
                predictions.append(probs.cpu())

        return torch.cat(predictions, dim=0)
