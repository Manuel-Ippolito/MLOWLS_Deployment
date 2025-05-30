import json
from pathlib import Path
from typing import Any

import timm
import torch

from ml_owls.model.config import Config
from interfaces.model.model_loader import ModelLoader
from ml_owls.model.persistence.model_saver import PyTorchModelSaver


class TrainedModelLoader(ModelLoader):
    """Loader for trained BirdCLEF models."""

    def __init__(self, config: Config):
        """Initialize model loader.

        Args:
            config: Training configuration
        """
        self.config = config
        print(f"📂 Model Loader initialized for {config.backbone}")

    def load_model(self, model_path: str, **kwargs: Any) -> torch.nn.Module:
        """Load trained model from checkpoint."""
        print(f"📂 Loading model: {model_path}")

        # Load metadata if available
        metadata_path = model_path.replace(".pth", "_metadata.json")
        if Path(metadata_path).exists():
            with open(metadata_path, "r") as f:
                metadata = json.load(f)
            print(f"📋 Model from epoch {metadata.get('epoch', 'unknown')}")

        # Create model architecture
        model = timm.create_model(
            self.config.backbone,
            pretrained=False,
            num_classes=self.config.num_classes,
            in_chans=1,
            drop_rate=self.config.dropout,
        )

        # Load weights
        saver = PyTorchModelSaver()
        model = saver.load_model(model, model_path)

        print(f"✅ Model loaded: {self.config.backbone} ({self.config.num_classes} classes)")
        return model
