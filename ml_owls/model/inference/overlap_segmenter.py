from typing import Any

import numpy as np

from interfaces.model.audo_segmenter import AudioSegmenter


class OverlapSegmenter(AudioSegmenter):
    """Audio segmenter with configurable overlap."""

    def __init__(self, segment_length: float = 30.0, overlap: float = 0.5) -> None:
        """Initialize overlap segmenter."""
        self.segment_length = segment_length
        self.overlap = overlap

        print("🔪 Overlap Segmenter initialized")
        print(f"   Segment length: {segment_length}s")
        print(f"   Overlap: {overlap * 100:.0f}%")

    def segment(
        self, audio: np.ndarray, sample_rate: int, **kwargs: Any
    ) -> tuple[list[np.ndarray], list[float]]:
        """Segment audio with overlap."""
        segment_samples = int(self.segment_length * sample_rate)
        hop_samples = int(segment_samples * (1 - self.overlap))

        segments = []
        timestamps = []

        # Handle short audio files
        if len(audio) <= segment_samples:
            # Pad if necessary
            segment = np.pad(audio, (0, max(0, segment_samples - len(audio))))
            segments.append(segment)
            timestamps.append(0.0)
            return segments, timestamps

        # Generate overlapping segments
        for start in range(0, len(audio) - segment_samples + 1, hop_samples):
            end = start + segment_samples
            segment = audio[start:end]

            segments.append(segment)
            timestamps.append(start / sample_rate)

        print(f"🔪 Generated {len(segments)} segments with {self.overlap * 100:.0f}% overlap")
        return segments, timestamps
