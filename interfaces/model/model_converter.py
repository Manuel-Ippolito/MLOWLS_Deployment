from abc import ABC, abstractmethod
from typing import Any

import torch


class ModelConverter(ABC):
    """Abstract interface for model conversion."""

    @abstractmethod
    def convert(self, model: torch.nn.Module, output_path: str, **kwargs: Any) -> dict[str, Any]:
        """Convert model to target format."""
        pass
