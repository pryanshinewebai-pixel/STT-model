"""
Feature Extractor — Computes spectrograms and acoustic features from waveforms.

Responsibility: Person 1 (Data Collection + Cleaning + Preprocessing)
"""

from typing import Union

import numpy as np
import torch
import torchaudio.transforms as T


class FeatureExtractor:
    """Extracts Mel spectrograms, MFCCs, and other acoustic features.

    Attributes:
        n_mels: Number of Mel filter banks.
        n_fft: FFT window size.
        hop_length: Hop length between frames.
        win_length: Window length for STFT.
        f_min: Minimum frequency for Mel filter bank.
        f_max: Maximum frequency for Mel filter bank.
        normalize: Whether to normalize features (CMVN).
    """

    def __init__(
        self,
        n_mels: int = 80,
        n_fft: int = 400,
        hop_length: int = 160,
        win_length: int = 400,
        f_min: float = 0.0,
        f_max: float = 8000.0,
        normalize: bool = True,
    ) -> None:
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.f_min = f_min
        self.f_max = f_max
        self.normalize = normalize

    def compute_mel_spectrogram(
        self, waveform: Union[np.ndarray, torch.Tensor], sample_rate: int = 16000
    ) -> torch.Tensor:
        """Compute log-Mel spectrogram from a waveform.

        Args:
            waveform: Audio waveform as 1D numpy array or torch Tensor.
            sample_rate: Sample rate of the waveform.

        Returns:
            Log-Mel spectrogram tensor of shape (n_mels, time_steps).
        """
        if isinstance(waveform, np.ndarray):
            waveform_tensor = torch.from_numpy(waveform).float()
        else:
            waveform_tensor = waveform.float()

        if waveform_tensor.dim() == 1:
            waveform_tensor = waveform_tensor.unsqueeze(0)

        mel_transform = T.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=self.n_fft,
            win_length=self.win_length,
            hop_length=self.hop_length,
            n_mels=self.n_mels,
            f_min=self.f_min,
            f_max=self.f_max,
        )

        mel_spec = mel_transform(waveform_tensor)
        log_mel_spec = torch.log(torch.clamp(mel_spec, min=1e-9)).squeeze(0)

        if self.normalize:
            log_mel_spec = self.normalize_features(log_mel_spec)

        return log_mel_spec

    def compute_mfcc(
        self,
        waveform: Union[np.ndarray, torch.Tensor],
        sample_rate: int = 16000,
        n_mfcc: int = 13,
    ) -> torch.Tensor:
        """Compute MFCCs from a waveform.

        Args:
            waveform: Audio waveform as 1D numpy array or torch Tensor.
            sample_rate: Sample rate of the waveform.
            n_mfcc: Number of MFCC coefficients to extract.

        Returns:
            MFCC features tensor of shape (n_mfcc, time_steps).
        """
        if isinstance(waveform, np.ndarray):
            waveform_tensor = torch.from_numpy(waveform).float()
        else:
            waveform_tensor = waveform.float()

        if waveform_tensor.dim() == 1:
            waveform_tensor = waveform_tensor.unsqueeze(0)

        mfcc_transform = T.MFCC(
            sample_rate=sample_rate,
            n_mfcc=n_mfcc,
            melkwargs={
                "n_fft": self.n_fft,
                "win_length": self.win_length,
                "hop_length": self.hop_length,
                "n_mels": self.n_mels,
                "f_min": self.f_min,
                "f_max": self.f_max,
            },
        )

        mfccs = mfcc_transform(waveform_tensor).squeeze(0)

        if self.normalize:
            mfccs = self.normalize_features(mfccs)

        return mfccs

    def normalize_features(
        self, features: Union[np.ndarray, torch.Tensor]
    ) -> Union[np.ndarray, torch.Tensor]:
        """Apply per-channel mean and variance normalization (CMVN).

        Args:
            features: Feature matrix of shape (channels, time_steps).

        Returns:
            Normalized feature matrix.
        """
        if isinstance(features, torch.Tensor):
            mean = features.mean(dim=-1, keepdim=True)
            std = features.std(dim=-1, keepdim=True) + 1e-8
            return (features - mean) / std
        else:
            mean = np.mean(features, axis=-1, keepdims=True)
            std = np.std(features, axis=-1, keepdims=True) + 1e-8
            return (features - mean) / std
