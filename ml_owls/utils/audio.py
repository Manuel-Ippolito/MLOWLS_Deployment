# app/utils/audio.py
import librosa
import numpy as np
import io

def preprocess_audio(file_bytes: io.BytesIO, sample_rate: int = 32000):
    y, _ = librosa.load(file_bytes, sr=sample_rate)
    mel = librosa.feature.melspectrogram(y=y, sr=sample_rate)
    log_mel = librosa.power_to_db(mel)
    # Normalize and add batch/channel dims: [1, 1, freq, time]
    return np.expand_dims(np.expand_dims(log_mel, axis=0), axis=0).astype(np.float32)
