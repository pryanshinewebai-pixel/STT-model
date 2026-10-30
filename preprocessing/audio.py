"""
Audio Processor — Handles audio loading, resampling, and normalization.

Responsibility: Person 1 (Data Collection + Cleaning + Preprocessing)
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import librosa
import numpy as np
import torch
import torchaudio
import torchaudio.transforms as T


class AudioProcessor:
    """Loads, resamples, normalizes, and validates raw audio files.

    Attributes:
        sample_rate: Target sample rate for all audio (default 16000 Hz).
        mono: Whether to convert all audio to mono.
        max_duration: Maximum allowed duration in seconds.
        min_duration: Minimum allowed duration in seconds.
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        mono: bool = True,
        max_duration: float = 300.0,
        min_duration: float = 0.1,
        trim_silence: bool = True,
    ) -> None:
        self.sample_rate = sample_rate
        self.mono = mono
        self.max_duration = max_duration
        self.min_duration = min_duration
        self.trim_silence_flag = trim_silence

    def load(self, file_path: str | Path) -> Tuple[np.ndarray, int]:
        """Load an audio file and return (waveform, sample_rate).

        Args:
            file_path: Path to the audio file (.wav, .mp3, .flac).

        Returns:
            Tuple of (waveform as 1D numpy float32 array, target sample rate).

        Raises:
            FileNotFoundError: If the audio file does not exist.
            ValueError: If audio duration is out of allowed bounds.
        """
        path = Path(file_path)
        if not path.is_file():
            raise FileNotFoundError(f"Audio file not found: {path}")

        try:
            waveform_tensor, orig_sr = torchaudio.load(str(path))
        except Exception:
            # Fallback to librosa for formats unsupported by default torchaudio backend
            waveform_np, orig_sr = librosa.load(str(path), sr=None, mono=self.mono)
            waveform_tensor = torch.from_numpy(waveform_np).unsqueeze(0)

        # Convert to mono if multi-channel
        if self.mono and waveform_tensor.shape[0] > 1:
            waveform_tensor = torch.mean(waveform_tensor, dim=0, keepdim=True)

        waveform = waveform_tensor.squeeze(0).numpy().astype(np.float32)

        # Resample if sample rates differ
        if orig_sr != self.sample_rate:
            waveform = self.resample(waveform, orig_sr)

        # Trim leading and trailing silence
        if self.trim_silence_flag:
            waveform = self.trim_silence(waveform)

        # Normalize amplitude
        waveform = self.normalize(waveform)

        # Validate duration
        if not self.validate(waveform, self.sample_rate):
            duration = len(waveform) / self.sample_rate
            raise ValueError(
                f"Audio duration ({duration:.2f}s) outside range [{self.min_duration}s, {self.max_duration}s]: {path}"
            )

        return waveform, self.sample_rate

    def resample(self, waveform: np.ndarray, orig_sr: int) -> np.ndarray:
        """Resample waveform to the target sample rate.

        Args:
            waveform: Input audio waveform (1D numpy array).
            orig_sr: Original sample rate of the waveform.

        Returns:
            Resampled waveform as 1D numpy array.
        """
        if orig_sr == self.sample_rate:
            return waveform

        waveform_tensor = torch.from_numpy(waveform).unsqueeze(0)
        resampler = T.Resample(orig_freq=orig_sr, new_freq=self.sample_rate)
        resampled_tensor = resampler(waveform_tensor)
        return resampled_tensor.squeeze(0).numpy().astype(np.float32)

    def normalize(self, waveform: np.ndarray) -> np.ndarray:
        """Normalize waveform amplitude to [-1.0, 1.0].

        Args:
            waveform: Input audio waveform.

        Returns:
            Normalized waveform.
        """
        max_val = np.max(np.abs(waveform))
        if max_val > 0:
            return (waveform / max_val).astype(np.float32)
        return waveform.astype(np.float32)

    def validate(self, waveform: np.ndarray, sample_rate: int) -> bool:
        """Check if audio meets duration and quality constraints.

        Args:
            waveform: Audio waveform.
            sample_rate: Sample rate of the waveform.

        Returns:
            True if the audio is valid, False otherwise.
        """
        duration = len(waveform) / sample_rate
        return self.min_duration <= duration <= self.max_duration

    def trim_silence(
        self, waveform: np.ndarray, top_db: float = 20.0
    ) -> np.ndarray:
        """Trim leading and trailing silence from the waveform.

        Args:
            waveform: Input audio waveform.
            top_db: Threshold (in dB) below peak to consider as silence.

        Returns:
            Trimmed waveform.
        """
        trimmed_waveform, _ = librosa.effects.trim(waveform, top_db=top_db)
        return trimmed_waveform.astype(np.float32)
