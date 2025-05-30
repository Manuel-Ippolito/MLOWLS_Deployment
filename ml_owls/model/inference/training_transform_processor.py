import os
import tempfile
from typing import Any

import numpy as np
import torch

from ml_owls.model.config import Config
from ml_owls.model.dataset import BirdClefDataset
from interfaces.model.audio_processor import AudioProcessor
from ml_owls.model.inference.overlap_segmenter import OverlapSegmenter


class TrainingTransformProcessor(AudioProcessor):
    """Audio processor using the training dataset logic."""

    def __init__(self, config: Config):
        """Initialize with training config."""
        self.config = config

        # Create dataset instance in inference mode
        self.dataset = BirdClefDataset(
            data_source=None,  # No CSV needed for inference
            audio_dir="",  # Not used for inference
            config=config,
            transform=None,  # Use default from config
            is_train=False,  # No training augmentations
            inference_mode=True,  # Skip CSV processing
        )

        # Create segmenter using dataset properties
        self.segmenter = OverlapSegmenter(
            segment_length=self.dataset.segment_length, overlap=self.dataset.overlap
        )

        print("🎼 Training Transform Processor initialized")
        print(f"   Sample rate: {self.dataset.sample_rate}Hz")
        print(f"   Segment length: {self.dataset.segment_length}s")
        print(f"   N_FFT: {self.dataset.n_fft}")
        print(f"   Hop length: {self.dataset.hop_length}")

    def process_audio_file(self, audio_path: str) -> np.ndarray:
        """Process audio file - delegates to process_file and returns only spectrograms."""
        spectrograms, _ = self.process_file(audio_path)
        return spectrograms

    def _to_inference_format(self, spectrogram: torch.Tensor) -> np.ndarray:
        """Convert to inference format."""
        spec_np = spectrogram.numpy() if hasattr(spectrogram, "numpy") else spectrogram

        # Ensure shape is (1, n_mels, n_frames) for ONNX
        if spec_np.ndim == 2:
            spec_np = spec_np[np.newaxis, :, :]

        return spec_np

    def process_file(self, audio_path: str, **kwargs: Any) -> tuple[np.ndarray, list[float]]:
        """Process file - return spectrograms and timestamps."""
        print(f"🔧 process_file called for: {audio_path}")

        audio = self.dataset.load_audio_for_inference(audio_path)
        print(f"🔧 Audio loaded: shape={audio.shape}, type={type(audio)}")

        # Get both segments and timestamps from segmenter (ONLY ONCE!)
        segments, timestamps = self.segmenter.segment(audio.numpy(), self.dataset.sample_rate)
        print(f"🔪 Generated {len(segments)} segments with 50% overlap")

        spectrograms = []
        for i, segment in enumerate(segments):
            print(f"🔧 Processing segment {i + 1}/{len(segments)}: shape={segment.shape}")
            segment_tensor = torch.from_numpy(segment).float()

            print(f"🔧 Calling dataset.transform_audio_segment for segment {i + 1}")
            spectrogram = self.dataset.transform_audio_segment(segment_tensor)
            print(f"🔧 Transform completed for segment {i + 1}: shape={spectrogram.shape}")

            spec_np = self._to_inference_format(spectrogram)
            spectrograms.append(spec_np)

        # Stack all spectrograms into a single batch
        batched_spectrograms = np.stack(spectrograms, axis=0)

        print(
            f"🔧 process_file returning: spectrograms {batched_spectrograms.shape}, timestamps {len(timestamps)}"
        )
        return batched_spectrograms, timestamps

    def process_bytes(self, audio_bytes: bytes, **kwargs: Any) -> tuple[np.ndarray, list[float]]:
        """Process audio bytes - return spectrograms and timestamps."""

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
            tmp_file.write(audio_bytes)
            tmp_path = tmp_file.name

        try:
            return self.process_file(tmp_path)
        finally:
            os.unlink(tmp_path)
