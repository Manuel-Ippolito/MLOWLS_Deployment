import numpy as np
from sklearn.metrics import roc_auc_score

from interfaces.model.metric_calculator import MetricCalculator


class AUCCalculator(MetricCalculator):
    """Calculate AUC metric with proper handling of missing classes."""

    def __init__(self, num_classes: int, average: str = "macro"):
        """Initialize AUC calculator.

        Args:
            num_classes: Total number of classes in the dataset
            average: Averaging strategy for multi-class ('macro', 'micro', 'weighted')
        """
        self.num_classes = num_classes
        self.average = average

    @property
    def name(self) -> str:
        return "auc"

    def calculate(self, predictions: np.ndarray, labels: np.ndarray) -> float:
        """Calculate AUC with proper probability renormalization.

        Args:
            predictions: Model predictions array of shape (n_samples, n_classes)
            labels: True labels array of shape (n_samples,)

        Returns:
            AUC score (0-1)
        """
        try:
            unique_labels = np.unique(labels)

            # Need at least 2 classes for AUC calculation
            if len(unique_labels) < 2:
                return 0.0

            # Binary classification case
            if self.num_classes == 2:
                return float(roc_auc_score(labels, predictions[:, 1]))

            # Multi-class: extract only relevant predictions and renormalize
            relevant_preds = predictions[:, unique_labels]

            # Renormalize to ensure probabilities sum to 1.0
            relevant_preds = relevant_preds / relevant_preds.sum(axis=1, keepdims=True)

            # Remap labels to match reduced prediction space
            label_mapping = {old_label: new_idx for new_idx, old_label in enumerate(unique_labels)}
            remapped_labels = np.array([label_mapping[label] for label in labels])

            return float(
                roc_auc_score(
                    remapped_labels, relevant_preds, multi_class="ovr", average=self.average
                )
            )

        except Exception:
            # Return 0.0 if AUC calculation fails for any reason
            return 0.0
