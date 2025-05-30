"""
Abstract model saver interface.
"""

from abc import ABC, abstractmethod
from typing import Any

import torch


class ModelSaver(ABC):
    """Abstract interface for saving models."""

    @abstractmethod
    def save_model(self, model: torch.nn.Module, path: str, metadata: dict[str, Any]) -> None:
        """Save model to specified path with metadata.

        Args:
            model: PyTorch model to save
            path: File path to save model
            metadata: Additional metadata to save with model
        """
        pass

    @abstractmethod
    def load_model(self, model: torch.nn.Module, path: str) -> torch.nn.Module:
        """Load model from specified path.

        Args:
            model: PyTorch model instance (for architecture)
            path: File path to load model from

        Returns:
            Loaded model
        """
        pass
