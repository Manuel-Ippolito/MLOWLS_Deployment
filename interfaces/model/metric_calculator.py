"""
Abstract metric calculator interface.
"""

from abc import ABC, abstractmethod

import numpy as np


class MetricCalculator(ABC):
    """Abstract interface for metric calculations."""

    @abstractmethod
    def calculate(self, predictions: np.ndarray, labels: np.ndarray) -> float:
        """Calculate metric from predictions and labels.

        Args:
            predictions: Model predictions array of shape (n_samples, n_classes)
            labels: True labels array of shape (n_samples,)

        Returns:
            Calculated metric value
        """
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Get metric name."""
        pass
