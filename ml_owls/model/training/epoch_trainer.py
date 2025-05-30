from typing import Dict, Optional

import numpy as np
import torch
import torch.nn.functional as F
from rich.console import Console
from rich.progress import BarColumn, Progress, SpinnerColumn, TimeElapsedColumn, TimeRemainingColumn

from interfaces.model.training_strategy import TrainingStrategy
from ml_owls.model.metrics.aggregator import MetricAggregator


class EpochTrainer:
    """Handles single epoch training/validation following Single Responsibility Principle."""

    def __init__(
        self,
        device: torch.device,
        strategy: TrainingStrategy,
        metric_aggregator: MetricAggregator,
        console: Console,
    ):
        """Initialize epoch trainer.

        Args:
            device: Device to run computations on
            strategy: Training strategy to use
            metric_aggregator: Aggregator for calculating metrics
            console: Rich console for progress display
        """
        self.device = device
        self.strategy = strategy
        self.metric_aggregator = metric_aggregator
        self.console = console

    def train_epoch(
        self,
        model: torch.nn.Module,
        dataloader: torch.utils.data.DataLoader,
        optimizer: torch.optim.Optimizer,
        criterion: torch.nn.Module,
        epoch: int,
    ) -> Dict[str, float]:
        """Train model for one epoch."""
        model.train()
        total_loss = 0.0
        all_predictions = []
        all_labels = []

        with Progress(
            SpinnerColumn(),
            "[progress.description]{task.description}",
            BarColumn(bar_width=None),
            "[progress.percentage]{task.percentage:>3.0f}%",
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            console=self.console,
        ) as progress:
            task = progress.add_task(f"[green]Epoch {epoch} Train", total=len(dataloader))

            for batch_idx, batch in enumerate(dataloader):
                # Execute training step using strategy
                loss, predictions, original_labels = self.strategy.train_step(
                    model, batch, optimizer, criterion, self.device
                )

                total_loss += loss.item()

                # Collect predictions and labels for metrics
                with torch.no_grad():
                    probs = F.softmax(predictions, dim=1)
                    all_predictions.extend(probs.cpu().numpy())
                    all_labels.extend(original_labels.cpu().numpy().tolist())

                # Update progress with live metrics every N batches for performance
                if batch_idx % 10 == 0 and all_predictions:
                    current_metrics = self._get_live_metrics(
                        all_predictions, all_labels, total_loss, batch_idx + 1
                    )
                    progress.update(
                        task,
                        advance=1,
                        description=f"[green]Epoch {epoch} Train — {current_metrics}",
                    )
                else:
                    progress.update(task, advance=1)

        predictions_array = np.array(all_predictions)
        labels_array = np.array(all_labels)

        metrics = self.metric_aggregator.calculate_all(predictions_array, labels_array)
        metrics["loss"] = total_loss / len(dataloader)

        metrics_str = ", ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
        self.console.log(f"[bold green]Epoch {epoch} Train ⏩ {metrics_str}")

        return metrics

    def validate_epoch(
        self,
        model: torch.nn.Module,
        dataloader: Optional[torch.utils.data.DataLoader],
        criterion: torch.nn.Module,
        epoch: int,
    ) -> Optional[Dict[str, float]]:
        """Validate model for one epoch."""
        if dataloader is None:
            return None

        model.eval()
        total_loss = 0.0
        all_predictions = []
        all_labels = []

        with Progress(
            SpinnerColumn(),
            "[progress.description]{task.description}",
            BarColumn(bar_width=None),
            "[progress.percentage]{task.percentage:>3.0f}%",
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            console=self.console,
        ) as progress:
            task = progress.add_task(f"[cyan]Epoch {epoch} Val", total=len(dataloader))

            with torch.no_grad():
                for batch_idx, batch in enumerate(dataloader):
                    specs, labels = batch
                    specs, labels = specs.to(self.device), labels.to(self.device)

                    predictions = model(specs)
                    loss = criterion(predictions, labels)

                    total_loss += loss.item()

                    probs = F.softmax(predictions, dim=1)
                    all_predictions.extend(probs.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy().tolist())

                    if batch_idx % 10 == 0 and all_predictions:
                        current_metrics = self._get_live_metrics(
                            all_predictions, all_labels, total_loss, batch_idx + 1
                        )
                        progress.update(
                            task,
                            advance=1,
                            description=f"[cyan]Epoch {epoch} Val — {current_metrics}",
                        )
                    else:
                        progress.update(task, advance=1)

        predictions_array = np.array(all_predictions)
        labels_array = np.array(all_labels)

        metrics = self.metric_aggregator.calculate_all(predictions_array, labels_array)
        metrics["loss"] = total_loss / len(dataloader)

        metrics_str = ", ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
        self.console.log(f"[bold cyan]Epoch {epoch} Val ⏩ {metrics_str}")

        return metrics

    def _get_live_metrics(
        self, predictions: list, labels: list, total_loss: float, num_batches: int
    ) -> str:
        """Get live metrics string for progress display.

        Args:
            predictions: List of prediction arrays
            labels: List of labels
            total_loss: Accumulated loss
            num_batches: Number of processed batches

        Returns:
            Formatted metrics string
        """
        if not predictions or not labels:
            return f"loss: {total_loss / num_batches:.4f}"

        try:
            pred_array = np.array(predictions)
            label_array = np.array(labels)
            live_metrics = self.metric_aggregator.calculate_all(pred_array, label_array)
            live_metrics["loss"] = total_loss / num_batches

            # Format for display (shorter names for progress bar)
            formatted_parts = []
            for key, value in live_metrics.items():
                if key == "accuracy":
                    formatted_parts.append(f"acc: {value:.1f}%")
                elif key == "auc":
                    formatted_parts.append(f"auc: {value:.3f}")
                else:
                    formatted_parts.append(f"{key}: {value:.4f}")

            return ", ".join(formatted_parts)

        except Exception:
            return f"loss: {total_loss / num_batches:.4f}"
