import torch
import torch.nn as nn
import torchaudio.transforms as T


def normalize_waveform(waveform: torch.Tensor, eps: float = 1e-9) -> torch.Tensor:
    """
    Normalize a waveform to zero mean and unit variance per sample.

    Args:
        waveform: Audio tensor of shape (channels, samples).
        eps: Small constant to avoid division by zero.

    Returns:
        Normalized waveform.
    """
    mean = waveform.mean(dim=-1, keepdim=True)
    std = waveform.std(dim=-1, keepdim=True)
    return (waveform - mean) / (std + eps)


def get_mel_log_transform(
    sample_rate: int = 32000,
    n_fft: int = 1024,
    hop_length: int = 320,
    n_mels: int = 224,
    fmin: float = 20.0,
    fmax: float = 16000.0,
    power: float = 2.0,
) -> torch.nn.Module:
    """
    Create a sequential transform pipeline that computes a log-mel spectrogram
    from waveform.

    Args:
        sample_rate: Sampling rate of the audio.
        n_fft: FFT window size.
        hop_length: Number of audio samples between STFT columns.
        n_mels: Number of mel frequency bins.
        fmin: Minimum frequency (Hz).
        fmax: Maximum frequency (Hz).
        power: Exponent for the magnitude spectrogram.

    Returns:
        A Torch Sequential module that applies MelSpectrogram and AmplitudeToDB.
    """
    mel_spectrogram = T.MelSpectrogram(
        sample_rate=sample_rate,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
        f_min=fmin,
        f_max=fmax,
        power=power,
        normalized=True,  # Add normalization for better stability
    )
    amplitude_to_db = T.AmplitudeToDB(stype="power", top_db=80)  # Limit dynamic range
    return torch.nn.Sequential(mel_spectrogram, amplitude_to_db)


def get_spectrogram_augmentations(is_train: bool = True) -> torch.nn.Module:
    """
    Create spectrogram augmentations for training robustness.

    Args:
        is_train: Whether to apply augmentations (training mode).

    Returns:
        Augmentation pipeline.
    """
    if not is_train:
        return nn.Identity()

    return nn.Sequential(T.FrequencyMasking(freq_mask_param=30), T.TimeMasking(time_mask_param=40))


def mixup_data(
    x: torch.Tensor, y: torch.Tensor, alpha: float = 1.0
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """
    Mixup data augmentation for robust training with weak labels.

    Args:
        x: Input batch.
        y: Label batch.
        alpha: Mixup strength parameter.

    Returns:
        Mixed inputs, label pairs, and lambda.
    """
    if alpha > 0:
        lam = torch.distributions.Beta(alpha, alpha).sample()
    else:
        lam = 1.0

    batch_size = x.size(0)
    index = torch.randperm(batch_size, device=x.device)

    mixed_x = lam * x + (1 - lam) * x[index]
    y_a, y_b = y, y[index]

    return mixed_x, y_a, y_b, lam


def normalize_spectrogram(melspec: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    Normalize mel spectrogram to zero mean and unit variance.

    Args:
        melspec: Mel spectrogram tensor.
        eps: Small constant to avoid division by zero.

    Returns:
        Normalized spectrogram.
    """
    mean = melspec.mean()
    std = melspec.std()
    return (melspec - mean) / (std + eps)
