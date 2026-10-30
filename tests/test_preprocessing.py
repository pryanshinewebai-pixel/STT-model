"""
Unit tests for AudioProcessor and FeatureExtractor in the preprocessing module.
"""

import numpy as np
import pytest
import torch

from preprocessing.audio import AudioProcessor
from preprocessing.features import FeatureExtractor


def test_audio_processor_normalize():
    processor = AudioProcessor()
    dummy_waveform = np.array([0.5, -1.5, 0.0, 2.0], dtype=np.float32)
    normalized = processor.normalize(dummy_waveform)

    assert np.max(np.abs(normalized)) == pytest.approx(1.0)
    assert normalized.shape == dummy_waveform.shape


def test_audio_processor_validate():
    processor = AudioProcessor(min_duration=0.5, max_duration=5.0)
    sr = 16000

    # 1 second audio - valid
    valid_waveform = np.zeros(sr, dtype=np.float32)
    assert processor.validate(valid_waveform, sr) is True

    # 0.1 second audio - invalid (too short)
    short_waveform = np.zeros(int(sr * 0.1), dtype=np.float32)
    assert processor.validate(short_waveform, sr) is False


def test_feature_extractor_mel_spectrogram():
    extractor = FeatureExtractor(n_mels=80, n_fft=400, hop_length=160, win_length=400)
    sample_rate = 16000
    duration_sec = 1.0
    waveform = np.sin(2 * np.pi * 440 * np.linspace(0, duration_sec, int(sample_rate * duration_sec)))

    mel_spec = extractor.compute_mel_spectrogram(waveform, sample_rate=sample_rate)

    assert isinstance(mel_spec, torch.Tensor)
    assert mel_spec.dim() == 2
    assert mel_spec.shape[0] == 80  # n_mels
    assert mel_spec.shape[1] > 0    # time steps


def test_feature_extractor_cmvn():
    extractor = FeatureExtractor(n_mels=80)
    dummy_features = torch.randn(80, 100)
    normalized = extractor.normalize_features(dummy_features)

    assert normalized.shape == dummy_features.shape
    # Means per mel channel should be approximately 0
    assert torch.allclose(normalized.mean(dim=-1), torch.zeros(80), atol=1e-5)
