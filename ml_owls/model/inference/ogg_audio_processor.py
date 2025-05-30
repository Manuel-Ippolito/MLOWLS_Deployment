import io
from pathlib import Path
from typing import Any

import librosa
import numpy as np

from interfaces.model.audio_processor import AudioProcessor
from interfaces.model.audo_segmenter import AudioSegmenter
from interfaces.model.spectrogram_generator import SpectrogramGenerator


class OGGAudioProcessor(AudioProcessor):
    """OGG audio processor with dependency injection."""

    def __init__(
        self,
        segmenter: AudioSegmenter,
        spectrogram_generator: SpectrogramGenerator,
        target_sample_rate: int = 32000,
    ) -> None:
        """Initialize OGG processor."""
        self.segmenter = segmenter
        self.spectrogram_generator = spectrogram_generator
        self.target_sample_rate = target_sample_rate

        print("🎵 OGG Audio Processor initialized")
        print(f"   Target sample rate: {target_sample_rate}Hz")
        print(f"   Segmenter: {type(segmenter).__name__}")
        print(f"   Spectrogram: {type(spectrogram_generator).__name__}")

    def process_file(self, audio_path: str, **kwargs: Any) -> tuple[list[np.ndarray], list[float]]:
        """Process OGG audio file."""
        try:
            # Load OGG audio
            audio, sr = librosa.load(audio_path, sr=self.target_sample_rate)
            print(f"📁 Loaded {Path(audio_path).name}: {len(audio) / sr:.1f}s @ {sr}Hz")

        except Exception as e:
            raise ValueError(f"Failed to load OGG file {audio_path}: {str(e)}")

        return self._process_audio_data(audio, sr, **kwargs)

    def process_bytes(
        self, audio_bytes: bytes, **kwargs: Any
    ) -> tuple[list[np.ndarray], list[float]]:
        """Process OGG audio from bytes."""
        try:
            # Load audio from bytes
            audio, sr = librosa.load(io.BytesIO(audio_bytes), sr=self.target_sample_rate)
            print(f"📦 Loaded audio from bytes: {len(audio) / sr:.1f}s @ {sr}Hz")

        except Exception as e:
            raise ValueError(f"Failed to load OGG from bytes: {str(e)}")

        return self._process_audio_data(audio, sr, **kwargs)

    def _process_audio_data(
        self, audio: np.ndarray, sample_rate: int, **kwargs: Any
    ) -> tuple[list[np.ndarray], list[float]]:
        """Process raw audio data."""
        # Resample if necessary
        if sample_rate != self.target_sample_rate:
            audio = librosa.resample(audio, orig_sr=sample_rate, target_sr=self.target_sample_rate)
            print(f"🔄 Resampled from {sample_rate}Hz to {self.target_sample_rate}Hz")

        # Segment audio
        segments, timestamps = self.segmenter.segment(audio, self.target_sample_rate, **kwargs)

        # Generate spectrograms
        spectrograms = []
        for segment in segments:
            # Generate spectrogram
            spec = self.spectrogram_generator.generate(segment)
            # Add channel dimension: (height, width) → (1, height, width)
            spec = spec[np.newaxis, :]
            spectrograms.append(spec)

        print(f"🎼 Generated {len(spectrograms)} spectrograms")
        return spectrograms, timestamps

    def get_metadata(self, audio_path: str) -> dict[str, Any]:
        """Get metadata from OGG file."""
        try:
            # Load just for metadata
            audio, sr = librosa.load(audio_path, sr=None)
            duration = len(audio) / sr

            return {
                "duration_seconds": duration,
                "original_sample_rate": sr,
                "samples": len(audio),
                "will_resample": sr != self.target_sample_rate,
                "target_sample_rate": self.target_sample_rate,
                "estimated_segments": self._estimate_segments(duration),
            }
        except Exception as e:
            return {"error": str(e)}

    def _estimate_segments(self, duration: float) -> int:
        """Estimate number of segments for given duration."""
        segment_length = getattr(self.segmenter, "segment_length", 30.0)
        overlap = getattr(self.segmenter, "overlap", 0.5)

        if duration <= segment_length:
            return 1

        hop_length = segment_length * (1 - overlap)
        return max(1, int((duration - segment_length) / hop_length) + 1)
