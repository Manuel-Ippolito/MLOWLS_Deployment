import logging
import os
from typing import Any, Optional, Union

import pandas as pd
import torch
import torch.nn.functional as F
import torchaudio
from rich.console import Console
from rich.progress import BarColumn, MofNCompleteColumn, Progress, SpinnerColumn, TimeElapsedColumn
from torch.utils.data import Dataset

from ml_owls.model.config import Config
from utils.model.utils import (
    get_mel_log_transform,
    get_spectrogram_augmentations,
    normalize_spectrogram,
    normalize_waveform,
)

logger = logging.getLogger(__name__)
console = Console()


def collate_fn(batch: list[tuple[torch.Tensor, torch.Tensor]]) -> tuple[torch.Tensor, torch.Tensor]:
    """Collate function for DataLoader."""
    specs, labels = zip(*batch)
    return torch.stack(specs), torch.stack(labels)


class BirdClefDataset(Dataset):
    """
    PyTorch Dataset for BirdCLEF with multiple overlapping segments per audio file.
    """

    def __init__(
        self,
        data_source: Union[pd.DataFrame, str, None],
        audio_dir: str,
        config: Config,
        transform: Optional[torch.nn.Module] = None,
        is_train: bool = True,
        inference_mode: bool = False,
    ) -> None:
        if inference_mode:
            self.df = None
            self.segments = []
        else:
            self.df = (
                data_source.copy()
                if isinstance(data_source, pd.DataFrame)
                else pd.read_csv(data_source)
            )
            assert "label_idx" in self.df.columns, "DataFrame must contain 'label_idx'."
            self.segments = self._create_segments()

        self.audio_dir = audio_dir
        self.config = config
        self.is_train = is_train

        self.segment_samples = int(self.config.sample_rate * self.config.segment_length)

        self.transform = transform or get_mel_log_transform(
            sample_rate=self.config.sample_rate,
            n_fft=self.config.n_fft,
            hop_length=self.config.hop_length,
            n_mels=self.config.n_mels,
            fmin=self.config.fmin,
            fmax=self.config.fmax,
            power=2.0,
        )

        self.spec_augment = get_spectrogram_augmentations(is_train=self.is_train)

    @property
    def sample_rate(self) -> int:
        """Get sample rate from config."""
        return getattr(self.config, "sample_rate", 32000)

    @property
    def segment_length(self) -> float:
        """Get segment length from config."""
        return getattr(self.config, "segment_length", 30.0)

    @property
    def overlap(self) -> float:
        """Get overlap from config."""
        return getattr(self.config, "overlap", 0.5)

    @property
    def n_fft(self) -> int:
        """Get n_fft from config."""
        return getattr(self.config, "n_fft", 1024)

    @property
    def hop_length(self) -> int:
        """Get hop_length from config."""
        return getattr(self.config, "hop_length", 320)

    @property
    def n_mels(self) -> int:
        """Get n_mels from config."""
        return getattr(self.config, "n_mels", 128)

    @property
    def fmin(self) -> float:
        """Get fmin from config."""
        return getattr(self.config, "fmin", 20.0)

    @property
    def fmax(self) -> float:
        """Get fmax from config."""
        return getattr(self.config, "fmax", 16000.0)

    def _create_segments(self) -> list[dict[str, Any]]:
        """Create multiple overlapping segments from each audio file."""
        segments: list[dict[str, Any]] = []
        hop_samples = int(self.segment_samples * (1 - self.config.overlap))

        if self.df is None:
            return segments

        console.log(
            f"[bold blue]Creating segments: {self.config.segment_length}s length, "
            f"{self.config.overlap} overlap, max {self.config.max_segments_per_file} per file"
        )

        with Progress(
            SpinnerColumn(),
            "[progress.description]{task.description}",
            BarColumn(bar_width=None),
            MofNCompleteColumn(),
            "[progress.percentage]{task.percentage:>3.0f}%",
            TimeElapsedColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("[green]Processing audio files", total=len(self.df))

            for idx, row in self.df.iterrows():
                filepath = os.path.join(self.audio_dir, row["filename"])

                try:
                    info = torchaudio.info(filepath)
                    audio_samples = info.num_frames
                except Exception as e:
                    logger.warning(f"Could not get info for {filepath}: {e}")
                    audio_samples = int(self.config.segment_length * self.config.sample_rate)

                segment_count = 0

                if audio_samples <= self.segment_samples:
                    segments.append(
                        {
                            "original_idx": idx,
                            "start_sample": 0,
                            "filename": row["filename"],
                            "primary_label": row["primary_label"],
                            "label_idx": row["label_idx"],
                        }
                    )
                    segment_count = 1
                else:
                    for start_sample in range(
                        0, audio_samples - self.segment_samples + 1, hop_samples
                    ):
                        if segment_count >= self.config.max_segments_per_file:
                            break

                        segments.append(
                            {
                                "original_idx": idx,
                                "start_sample": start_sample,
                                "filename": row["filename"],
                                "primary_label": row["primary_label"],
                                "label_idx": row["label_idx"],
                            }
                        )
                        segment_count += 1

                # Update progress with current stats
                progress.update(
                    task,
                    advance=1,
                    description=f"[green]Processing audio files — {len(segments)} segments created",
                )

        console.log(
            f"[bold green]✅ Created {len(segments)} segments from {len(self.df)} audio files "
            f"(avg {len(segments) / len(self.df):.1f} segments per file)"
        )

        return segments

    def __len__(self) -> int:
        """Return number of segments."""
        return len(self.segments)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Get a segment and its label."""
        segment_info = self.segments[idx]
        filepath = os.path.join(self.audio_dir, segment_info["filename"])

        try:
            waveform, sr = torchaudio.load(filepath)
        except Exception as e:
            logger.error(f"Failed to load audio file {filepath}: {e}")
            raise

        if sr != self.config.sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.config.sample_rate)
            waveform = resampler(waveform)

        start = segment_info["start_sample"]
        end = start + self.segment_samples

        if end <= waveform.size(1):
            waveform = waveform[:, start:end]
        else:
            waveform = waveform[:, start:]

        current_length = waveform.size(1)
        if current_length < self.segment_samples:
            pad_length = self.segment_samples - current_length
            waveform = F.pad(waveform, (0, pad_length))
        elif current_length > self.segment_samples:
            waveform = waveform[:, : self.segment_samples]

        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        waveform = normalize_waveform(waveform)

        # Random time shift augmentation for training
        if self.is_train and torch.rand(1) < 0.3:
            max_shift = min(int(0.1 * self.segment_samples), waveform.size(1) // 10)
            if max_shift > 0:
                shift = torch.randint(-max_shift, max_shift + 1, (1,)).item()
                if shift > 0:
                    waveform = F.pad(waveform[:, shift:], (0, shift))
                elif shift < 0:
                    waveform = F.pad(waveform[:, :shift], (-shift, 0))

        # Convert to mel spectrogram
        melspec = self.transform(waveform)

        # Convert to mono if needed
        if melspec.shape[0] > 1:
            melspec = melspec.mean(dim=0, keepdim=True)

        # Normalize spectrogram
        melspec = normalize_spectrogram(melspec)

        # Apply spectrogram augmentations for training
        if self.is_train:
            melspec = self.spec_augment(melspec)

        label = torch.tensor(segment_info["label_idx"], dtype=torch.long)

        return melspec, label

    def get_stats(self) -> dict[str, Union[int, float]]:
        """Get dataset statistics."""
        # Fix: Check if df is None
        if self.df is None:
            return {
                "original_files": 0,
                "total_segments": len(self.segments),
                "avg_segments_per_file": 0.0,
                "unique_classes": 0,
                "min_segments_per_class": 0,
                "max_segments_per_class": 0,
            }

        class_counts: dict[int, int] = {}
        for segment in self.segments:
            label = segment["label_idx"]
            class_counts[label] = class_counts.get(label, 0) + 1

        stats = {
            "original_files": len(self.df),
            "total_segments": len(self.segments),
            "avg_segments_per_file": len(self.segments) / len(self.df) if len(self.df) > 0 else 0.0,
            "unique_classes": len(class_counts),
            "min_segments_per_class": min(class_counts.values()) if class_counts else 0,
            "max_segments_per_class": max(class_counts.values()) if class_counts else 0,
        }

        logger.info(f"Dataset stats: {stats}")
        return stats

    def load_audio_for_inference(self, audio_path: str) -> torch.Tensor:
        """Load and preprocess audio for inference (without segmentation)."""
        try:
            waveform, sr = torchaudio.load(audio_path)
        except Exception as e:
            logger.error(f"Failed to load audio file {audio_path}: {e}")
            raise

        # Resample if needed
        if sr != self.config.sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.config.sample_rate)
            waveform = resampler(waveform)

        # Convert to mono
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        # Normalize
        waveform = normalize_waveform(waveform)

        return waveform.squeeze(0)  # Return as 1D tensor

    def transform_audio_segment(self, audio_segment: torch.Tensor) -> torch.Tensor:
        """Transform audio segment to spectrogram (for inference)."""
        # Ensure correct length
        if audio_segment.numel() < self.segment_samples:
            pad_length = self.segment_samples - audio_segment.numel()
            audio_segment = F.pad(audio_segment, (0, pad_length))
        elif audio_segment.numel() > self.segment_samples:
            audio_segment = audio_segment[: self.segment_samples]

        # Add channel dimension for transform
        audio_segment = audio_segment.unsqueeze(0)

        # Apply mel transform (same as training)
        melspec = self.transform(audio_segment)

        # Convert to mono if needed
        if melspec.shape[0] > 1:
            melspec = melspec.mean(dim=0, keepdim=True)

        # Normalize spectrogram
        melspec = normalize_spectrogram(melspec)

        # NO augmentations for inference
        # Return without channel dimension (n_mels, n_frames)
        return melspec.squeeze(0)
