"""
Abstract logger interface for defining logging contracts.
"""

from abc import ABC, abstractmethod
from typing import Any, Optional, Union


class Logger(ABC):
    """Abstract interface for logging experiment data."""

    @abstractmethod
    def log_metric(self, key: str, value: Union[int, float], step: Optional[int] = None) -> None:
        """Log a single metric value.

        Args:
            key: Metric name
            value: Metric value
            step: Optional step/epoch number
        """
        pass

    @abstractmethod
    def log_param(self, key: str, value: Union[str, int, float, bool]) -> None:
        """Log a parameter value.

        Args:
            key: Parameter name
            value: Parameter value
        """
        pass

    @abstractmethod
    def log_artifact(self, path: str, artifact_path: Optional[str] = None) -> None:
        """Log an artifact file.

        Args:
            path: Local file path
            artifact_path: Remote artifact path
        """
        pass

    @abstractmethod
    def log_config(self, config: Any) -> None:
        """Log configuration object.

        Args:
            config: Configuration object to log
        """
        pass

    @abstractmethod
    def log_batch_metrics(
        self, metrics_dict: dict[str, Union[int, float]], step: Optional[int] = None
    ) -> None:
        """Log multiple metrics at once.

        Args:
            metrics_dict: Dictionary of metric name -> value pairs
            step: Optional step number for time series tracking
        """
        pass
