"""
Abstract training strategy interface.
"""

from abc import ABC, abstractmethod
from typing import Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    import torch


class TrainingStrategy(ABC):
    """Abstract interface for training strategies."""

    @abstractmethod
    def train_step(
        self,
        model: 'torch.nn.Module',
        batch: Tuple['torch.Tensor', 'torch.Tensor'],
        optimizer: 'torch.optim.Optimizer',
        criterion: 'torch.nn.Module',
        device: 'torch.device',
    ) -> Tuple['torch.Tensor', 'torch.Tensor', 'torch.Tensor']:
        """Execute one training step.

        Args:
            model: Neural network model
            batch: Tuple of (inputs, labels)
            optimizer: Optimizer instance
            criterion: Loss function
            device: Device to run computation on

        Returns:
            Tuple of (loss, predictions, original_labels)
        """
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Get strategy name."""
        pass
