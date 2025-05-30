from abc import ABC, abstractmethod
from typing import Any

import numpy as np


class SpectrogramGenerator(ABC):
    """Abstract interface for generating spectrograms."""

    @abstractmethod
    def generate(self, audio: np.ndarray) -> np.ndarray:
        """Generate spectrogram from audio."""
        pass

    @abstractmethod
    def configure(self, **kwargs: Any) -> None:
        """Configure spectrogram parameters."""
        pass
