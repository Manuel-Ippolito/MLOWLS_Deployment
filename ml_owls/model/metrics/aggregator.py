import numpy as np

from interfaces.model.metric_calculator import MetricCalculator


class MetricAggregator:
    """Aggregates multiple metrics for batch calculation."""

    def __init__(self) -> None:
        """Initialize empty metric aggregator."""
        self.calculators: dict[str, MetricCalculator] = {}

    def add_metric(self, calculator: MetricCalculator) -> None:
        """Add a metric calculator.

        Args:
            calculator: Metric calculator instance
        """
        self.calculators[calculator.name] = calculator

    def remove_metric(self, name: str) -> None:
        """Remove a metric calculator by name.

        Args:
            name: Name of metric to remove
        """
        if name in self.calculators:
            del self.calculators[name]

    def calculate_all(self, predictions: np.ndarray, labels: np.ndarray) -> dict[str, float]:
        """Calculate all registered metrics.

        Args:
            predictions: Model predictions array of shape (n_samples, n_classes)
            labels: True labels array of shape (n_samples,)

        Returns:
            Dictionary mapping metric names to calculated values
        """
        return {
            name: calculator.calculate(predictions, labels)
            for name, calculator in self.calculators.items()
        }

    def get_metric_names(self) -> list[str]:
        """Get names of all registered metrics.

        Returns:
            List of metric names
        """
        return list(self.calculators.keys())

    def has_metric(self, name: str) -> bool:
        """Check if metric is registered.

        Args:
            name: Metric name to check

        Returns:
            True if metric is registered, False otherwise
        """
        return name in self.calculators
