"""
Factory for creating trainers with proper dependency injection.
"""

from typing import Any, Dict

import torch

from interfaces.model.logger import Logger
from interfaces.model.training_strategy import TrainingStrategy
from ml_owls.model.metrics.accuracy import AccuracyCalculator
from ml_owls.model.metrics.aggregator import MetricAggregator
from ml_owls.model.metrics.auc import AUCCalculator
from ml_owls.model.persistence.model_saver import PyTorchModelSaver
from ml_owls.model.training.strategies import (
    CutMixTrainingStrategy,
    MixupTrainingStrategy,
    StandardTrainingStrategy,
)
from ml_owls.model.training.trainer import Trainer


class TrainerFactory:
    """Factory for creating trainers following Open/Closed principle."""

    @staticmethod
    def create_trainer(
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        criterion: torch.nn.Module,
        dataloaders: Dict[str, torch.utils.data.DataLoader],
        device: torch.device,
        config: Any,
        logger: Logger,
    ) -> Trainer:
        """Create a fully configured trainer.

        Args:
            model: Neural network model
            optimizer: Optimizer instance
            criterion: Loss function
            dataloaders: Dictionary of train/val dataloaders
            device: Device to run computations on
            config: Configuration object
            logger: Logger implementation

        Returns:
            Configured trainer instance
        """
        # Create training strategy based on config
        strategy = TrainerFactory._create_training_strategy(config)

        # Create metrics aggregator
        metric_aggregator = TrainerFactory._create_metric_aggregator(config)

        # Create model saver
        model_saver = PyTorchModelSaver()

        return Trainer(
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            dataloaders=dataloaders,
            device=device,
            logger=logger,
            model_saver=model_saver,
            strategy=strategy,
            metric_aggregator=metric_aggregator,
            config=config,
        )

    @staticmethod
    def _create_training_strategy(config: Any) -> TrainingStrategy:
        """Create training strategy based on configuration.

        Args:
            config: Configuration object

        Returns:
            Training strategy instance
        """

        strategy_name = config.training.training_strategy.lower()
        if not strategy_name:
            strategy_name = "standard"
        if strategy_name == "mixup":
            alpha = getattr(config, "mixup_alpha", 0.4)
            return MixupTrainingStrategy(alpha=alpha)
        elif strategy_name == "cutmix":
            alpha = getattr(config, "cutmix_alpha", 1.0)
            return CutMixTrainingStrategy(alpha=alpha)
        else:
            return StandardTrainingStrategy()

    @staticmethod
    def _create_metric_aggregator(config: Any) -> MetricAggregator:
        """Create metrics aggregator with configured metrics.

        Args:
            config: Configuration object

        Returns:
            Configured metric aggregator
        """
        aggregator = MetricAggregator()

        # Always add accuracy
        top_k = getattr(config, "accuracy_top_k", 1)
        aggregator.add_metric(AccuracyCalculator(top_k=top_k))

        # Add AUC if we have more than 2 classes
        num_classes = getattr(config, "num_classes", 2)
        if num_classes >= 2:
            auc_average = getattr(config, "auc_average", "macro")
            aggregator.add_metric(AUCCalculator(num_classes=num_classes, average=auc_average))

        return aggregator


class TrainingStrategyFactory:
    """Separate factory for training strategies for better extensibility."""

    @staticmethod
    def create(strategy_name: str, **kwargs: Any) -> TrainingStrategy:
        """Create training strategy by name.

        Args:
            strategy_name: Name of strategy to create
            **kwargs: Strategy-specific parameters

        Returns:
            Training strategy instance

        Raises:
            ValueError: If strategy name is not recognized
        """
        strategy_name_lower = strategy_name.lower()

        if strategy_name_lower == "standard":
            return StandardTrainingStrategy()
        elif strategy_name_lower == "mixup":
            alpha = kwargs.get("alpha", 0.4)
            return MixupTrainingStrategy(alpha=alpha)
        elif strategy_name_lower == "cutmix":
            alpha = kwargs.get("alpha", 1.0)
            return CutMixTrainingStrategy(alpha=alpha)
        else:
            available = ["standard", "mixup", "cutmix"]
            raise ValueError(
                f"Unknown strategy '{strategy_name}'. Available: {', '.join(available)}"
            )

    @staticmethod
    def list_strategies() -> list[str]:
        """List available strategy names.

        Returns:
            List of strategy names
        """
        return ["standard", "mixup", "cutmix"]
