from abc import ABC, abstractmethod
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    import torch

class ModelLoader(ABC):
    """Abstract interface for loading models."""

    @abstractmethod
    def load_model(self, model_path: str, **kwargs: Any) -> 'torch.nn.Module':
        """Load model from path."""
        pass
