from abc import ABC, abstractmethod
from typing import Any

import numpy as np


class AudioSegmenter(ABC):
    """Abstract interface for audio segmentation."""

    @abstractmethod
    def segment(
        self, audio: np.ndarray, sample_rate: int, **kwargs: Any
    ) -> tuple[list[np.ndarray], list[float]]:
        """Segment audio into chunks."""
        pass
