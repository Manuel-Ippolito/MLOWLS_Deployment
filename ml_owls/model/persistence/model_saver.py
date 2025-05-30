"""
Concrete model saver implementation.
"""

import json
import os
from typing import Any

import torch

from interfaces.model.model_saver import ModelSaver


class PyTorchModelSaver(ModelSaver):
    """PyTorch implementation of model saver."""

    def save_model(self, model: torch.nn.Module, path: str, metadata: dict[str, Any]) -> None:
        """Save PyTorch model with metadata.

        Args:
            model: PyTorch model to save
            path: File path to save model
            metadata: Additional metadata to save with model
        """
        # Ensure directory exists
        os.makedirs(os.path.dirname(path), exist_ok=True)

        # Save model state dict
        torch.save(model.state_dict(), path)

        # Save metadata as JSON
        metadata_path = path.replace(".pth", "_metadata.json")
        with open(metadata_path, "w") as f:
            # Convert non-serializable values to strings
            serializable_metadata = {}
            for key, value in metadata.items():
                if isinstance(value, (str, int, float, bool, list, dict)):
                    serializable_metadata[key] = value
                else:
                    serializable_metadata[key] = str(value)

            json.dump(serializable_metadata, f, indent=2)

    def load_model(self, model: torch.nn.Module, path: str) -> torch.nn.Module:
        """Load PyTorch model from path.

        Args:
            model: PyTorch model instance (for architecture)
            path: File path to load model from

        Returns:
            Loaded model
        """
        state_dict = torch.load(path, map_location="cpu")
        model.load_state_dict(state_dict)
        return model

    def load_metadata(self, path: str) -> dict[str, Any]:
        """Load metadata for a saved model.

        Args:
            path: Model file path

        Returns:
            Metadata dictionary
        """
        metadata_path = path.replace(".pth", "_metadata.json")

        if os.path.exists(metadata_path):
            with open(metadata_path, "r") as f:
                metadata: dict[str, Any] = json.load(f)
                return metadata

        return {}
