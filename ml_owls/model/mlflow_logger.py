from typing import Any, Optional, Union

import mlflow
import mlflow.pytorch

from interfaces.model.logger import Logger


class MLflowLogger(Logger):
    """MLflow implementation of the Logger interface."""

    def __init__(self, experiment_name: str, run_name: Optional[str] = None) -> None:
        """Initialize MLflow logger.

        Args:
            experiment_name: Name of the MLflow experiment
            run_name: Optional name for the run
        """
        self.experiment_name = experiment_name
        self.run_name = run_name
        self._run = None

    def __enter__(self) -> "MLflowLogger":
        """Start MLflow run."""
        mlflow.set_experiment(self.experiment_name)
        self._run = mlflow.start_run(run_name=self.run_name)
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """End MLflow run."""
        mlflow.end_run()

    def log_metric(self, key: str, value: Union[int, float], step: Optional[int] = None) -> None:
        """Log a single metric value."""
        mlflow.log_metric(key, value, step=step)

    def log_param(self, key: str, value: Union[str, int, float, bool]) -> None:
        """Log a parameter value."""
        mlflow.log_param(key, value)

    def log_artifact(self, path: str, artifact_path: Optional[str] = None) -> None:
        """Log an artifact file."""
        mlflow.log_artifact(path, artifact_path=artifact_path)

    def log_config(self, config: Any) -> None:
        """Log configuration object."""
        if hasattr(config, "__dict__"):
            for key, value in config.__dict__.items():
                if not key.startswith("_"):
                    self.log_param(f"config_{key}", value)
        elif isinstance(config, dict):
            for key, value in config.items():
                self.log_param(f"config_{key}", value)

    def log_batch_metrics(
        self, metrics_dict: dict[str, Union[int, float]], step: Optional[int] = None
    ) -> None:
        """Log multiple metrics at once."""
        for key, value in metrics_dict.items():
            self.log_metric(key, value, step=step)

    def log_dataset_stats(self, stats: dict[str, Union[int, float]]) -> None:
        """Log dataset statistics."""
        for key, value in stats.items():
            self.log_param(f"dataset_{key}", value)

    def log_model(self, model: Any, artifact_path: str = "model") -> None:
        """Log PyTorch model."""
        mlflow.pytorch.log_model(model, artifact_path)
