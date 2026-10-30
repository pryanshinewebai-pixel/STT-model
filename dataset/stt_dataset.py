"""
STT Dataset & Data Collator for Proprietary Speech-to-Text Model.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import librosa
import numpy as np
import pandas as pd
import soundfile as sf
import torch
import torchaudio.transforms as T
from torch.utils.data import Dataset

from preprocessing.tokenizer import CTCTokenizer


class STTDataset(Dataset):
    """PyTorch Dataset for Proprietary STT Training.

    Loads audio files, extracts 80-channel log-Mel spectrograms,
    and converts text transcripts to CTC token IDs.
    """

    def __init__(
        self,
        manifest_path: str | Path,
        tokenizer: CTCTokenizer,
        sample_rate: int = 16000,
        n_mels: int = 80,
        max_duration: float = 25.0,
        min_duration: float = 0.5,
    ) -> None:
        self.manifest_path = Path(manifest_path)
        self.manifest_dir = self.manifest_path.parent
        self.tokenizer = tokenizer
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.max_duration = max_duration
        self.min_duration = min_duration

        self.mel_transform = T.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=400,
            win_length=400,
            hop_length=160,
            n_mels=n_mels,
            f_min=0.0,
            f_max=8000.0,
        )

        df = pd.read_csv(self.manifest_path)
        # Filter duration if present
        if "duration" in df.columns:
            df = df[(df["duration"] >= min_duration) & (df["duration"] <= max_duration)]

        # Filter empty transcripts
        df = df[df["transcript"].notna() & (df["transcript"].str.strip() != "")]
        self.samples = df.to_dict("records")

    def _resolve_audio_path(self, raw_path: str) -> str:
        raw_str = str(raw_path).strip()
        if os.path.exists(raw_str):
            return raw_str

        # Try relative to manifest
        fname = os.path.basename(raw_str)
        c1 = str(self.manifest_dir / "wav_chunks" / fname)
        if os.path.exists(c1):
            return c1

        c2 = str(self.manifest_dir / fname)
        if os.path.exists(c2):
            return c2

        return raw_str

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.samples[idx]
        audio_path = self._resolve_audio_path(row["audio_path"])
        transcript = str(row["transcript"]).strip()

        # Load audio (16kHz mono)
        try:
            audio, sr = sf.read(audio_path)
            if audio.ndim > 1:
                audio = audio.mean(axis=1)
            if sr != self.sample_rate:
                audio = librosa.resample(audio, orig_sr=sr, target_sr=self.sample_rate)
        except Exception:
            # Fallback to zeros on read error
            audio = np.zeros(self.sample_rate, dtype=np.float32)

        # Convert to tensor & compute log-Mel
        waveform = torch.from_numpy(audio).float()
        if waveform.abs().max() > 0:
            waveform = (waveform / waveform.abs().max()) * 0.95

        # Mel spectrogram shape: (1, n_mels, time) -> (n_mels, time)
        mel = self.mel_transform(waveform)
        log_mel = torch.log(torch.clamp(mel, min=1e-5))

        # Per-utterance normalization (CMVN)
        mean = log_mel.mean(dim=-1, keepdim=True)
        std = log_mel.std(dim=-1, keepdim=True) + 1e-5
        norm_mel = (log_mel - mean) / std

        # Tokenize
        token_ids = self.tokenizer.encode(transcript)

        return {
            "mel": norm_mel,  # (n_mels, time)
            "target": torch.tensor(token_ids, dtype=torch.long),
            "transcript": transcript,
            "language": str(row.get("language", "unknown")),
        }


def collate_fn_ctc(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Pads Mel spectrograms and token targets for CTCLoss computation."""
    batch_size = len(batch)
    mel_channels = batch[0]["mel"].shape[0]

    # Find max lengths in batch
    mel_lens = [item["mel"].shape[1] for item in batch]
    target_lens = [len(item["target"]) for item in batch]

    max_mel_len = max(mel_lens)
    max_target_len = max(max_target_len for max_target_len in target_lens if max_target_len > 0) if max(target_lens) > 0 else 1

    # Allocate padded tensors
    padded_mels = torch.zeros(batch_size, mel_channels, max_mel_len, dtype=torch.float32)
    padded_targets = torch.zeros(batch_size, max_target_len, dtype=torch.long)

    for i, item in enumerate(batch):
        time_len = mel_lens[i]
        padded_mels[i, :, :time_len] = item["mel"]

        tgt = item["target"]
        if len(tgt) > 0:
            padded_targets[i, : len(tgt)] = tgt

    return {
        "features": padded_mels,  # (batch, n_mels, max_time)
        "feature_lengths": torch.tensor(mel_lens, dtype=torch.long),
        "targets": padded_targets,  # (batch, max_target_len)
        "target_lengths": torch.tensor(target_lens, dtype=torch.long),
        "transcripts": [item["transcript"] for item in batch],
    }
