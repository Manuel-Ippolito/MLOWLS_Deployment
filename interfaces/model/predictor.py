from abc import ABC, abstractmethod

import numpy as np


class Predictor(ABC):
    """Abstract interface for model prediction."""

    @abstractmethod
    def predict(self, inputs: np.ndarray) -> np.ndarray:
        """Make predictions on input data.

        Args:
            inputs: Input data array

        Returns:
            Prediction probabilities
        """
        pass

    @abstractmethod
    def predict_batch(self, inputs: list[np.ndarray]) -> list[np.ndarray]:
        """Make predictions on batch of inputs.

        Args:
            inputs: List of input arrays

        Returns:
            List of prediction arrays
        """
        pass
