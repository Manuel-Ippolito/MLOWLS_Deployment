from typing import Any

import numpy as np
import torch

from interfaces.model.spectrogram_generator import SpectrogramGenerator
from utils.model.utils import get_mel_log_transform


class MelSpectrogramGenerator(SpectrogramGenerator):
    """Mel-spectrogram generator using EXACT training transform."""

    def __init__(
        self,
        sample_rate: int = 32000,
        n_mels: int = 128,
        n_fft: int = 1024,
        hop_length: int = 320,
        fmin: float = 20.0,
        fmax: float = 16000.0,
        power: float = 2.0,
        target_frames: int | None = None,
    ) -> None:
        """Initialize with EXACT training transform."""
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.fmin = fmin
        self.fmax = fmax
        self.power = power
        self.target_frames = target_frames

        # Use the EXACT same transform as training
        self.transform = get_mel_log_transform(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            fmin=fmin,
            fmax=fmax,
            power=power,
        )

        print("🎼 Mel-Spectrogram Generator initialized (using training transform)")
        print(f"   Sample rate: {sample_rate}Hz")
        print(f"   Mel bands: {n_mels}")
        print(f"   FFT size: {n_fft}")
        print(f"   Hop length: {hop_length}")
        print(f"   Frequency range: {fmin}-{fmax}Hz")
        if target_frames:
            print(f"   🎯 Target frames: {target_frames}")

    def generate(self, audio: np.ndarray) -> np.ndarray:
        """Generate mel-spectrogram using EXACT training transform."""
        # Convert to torch tensor (as in training)
        audio_tensor = torch.from_numpy(audio).float()

        # Apply the EXACT same transform as training
        mel_spec = self.transform(audio_tensor)

        # Ensure target frame size if specified
        if self.target_frames and mel_spec.shape[-1] != self.target_frames:
            current_frames = mel_spec.shape[-1]

            if current_frames > self.target_frames:
                # Crop to target size
                mel_spec = mel_spec[..., : self.target_frames]
                print(f"🔧 Cropped {current_frames} → {self.target_frames} frames")
            elif current_frames < self.target_frames:
                # Pad to target size
                pad_width = self.target_frames - current_frames
                mel_spec = torch.nn.functional.pad(
                    mel_spec, (0, pad_width), mode="constant", value=-80.0
                )
                print(f"🔧 Padded {current_frames} → {self.target_frames} frames")

        # Convert back to numpy and ensure correct shape
        # Training transform returns: (n_mels, n_frames)
        # Model expects: (1, n_mels, n_frames)
        mel_spec_np = mel_spec.numpy()

        if mel_spec_np.ndim == 2:
            # Add channel dimension: (n_mels, n_frames) → (1, n_mels, n_frames)
            mel_spec_np = mel_spec_np[np.newaxis, :, :]

        return mel_spec_np

    def configure(self, **kwargs: Any) -> None:
        """Configure parameters and recreate transform."""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
                print(f"🔧 Updated {key}: {value}")

        self.transform = get_mel_log_transform(
            sample_rate=self.sample_rate,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            n_mels=self.n_mels,
            fmin=self.fmin,
            fmax=self.fmax,
            power=self.power,
        )
