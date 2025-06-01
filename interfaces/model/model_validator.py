from abc import ABC, abstractmethod
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    import torch


class ModelValidator(ABC):
    """Abstract interface for model validation."""

    @abstractmethod
    def validate(
        self, original_model: 'torch.nn.Module', converted_model_path: str, test_input: 'torch.Tensor'
    ) -> dict[str, Any]:
        """Validate converted model against original.

        Args:
            original_model: Original PyTorch model
            converted_model_path: Path to converted model
            test_input: Test input tensor

        Returns:
            Validation results
        """
        pass
