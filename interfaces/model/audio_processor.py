from abc import ABC, abstractmethod
from typing import Any

import numpy as np


class AudioProcessor(ABC):
    """Abstract base class for audio processors."""

    @abstractmethod
    def process_file(self, audio_path: str, **kwargs: Any) -> tuple[np.ndarray, list[float]]:
        """Process audio file and return spectrograms and timestamps."""
        pass

    @abstractmethod
    def process_bytes(self, audio_bytes: bytes, **kwargs: Any) -> tuple[np.ndarray, list[float]]:
        """Process audio bytes and return spectrograms and timestamps."""
        pass
