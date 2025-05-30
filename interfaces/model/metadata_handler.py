from abc import ABC, abstractmethod
from typing import Any


class MetadataHandler(ABC):
    """Abstract interface for handling metadata."""

    @abstractmethod
    def save_metadata(self, metadata: dict[str, Any], path: str) -> None:
        """Save metadata to file.

        Args:
            metadata: Metadata dictionary
            path: Output file path
        """
        pass

    @abstractmethod
    def load_metadata(self, path: str) -> dict[str, Any]:
        """Load metadata from file.

        Args:
            path: Metadata file path

        Returns:
            Metadata dictionary
        """
        pass
