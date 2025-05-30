import numpy as np
from sklearn.metrics import accuracy_score

from interfaces.model.metric_calculator import MetricCalculator


class AccuracyCalculator(MetricCalculator):
    """Calculate accuracy metric with optional top-k support."""

    def __init__(self, top_k: int = 1) -> None:
        """Initialize accuracy calculator.

        Args:
            top_k: Calculate top-k accuracy (default: 1 for standard accuracy)
        """
        self.top_k = top_k

    @property
    def name(self) -> str:
        return f"accuracy_top{self.top_k}" if self.top_k > 1 else "accuracy"

    def calculate(self, predictions: np.ndarray, labels: np.ndarray) -> float:
        """Calculate accuracy from predictions and labels.

        Args:
            predictions: Model predictions array of shape (n_samples, n_classes)
            labels: True labels array of shape (n_samples,)

        Returns:
            Accuracy percentage (0-100)
        """
        if self.top_k == 1:
            predicted_classes = np.argmax(predictions, axis=1)
            return float(accuracy_score(labels, predicted_classes) * 100)
        else:
            # Top-k accuracy
            top_k_preds = np.argsort(predictions, axis=1)[:, -self.top_k :]
            correct = np.any(top_k_preds == labels.reshape(-1, 1), axis=1)
            return float(correct.mean() * 100)
