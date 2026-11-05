"""
Proprietary Speech-To-Text Inference Pipeline
Loads trained Transformer-CTC model and Multilingual Language Model
for real-time speech transcription on CPU, MPS (Apple Silicon), or CUDA.
"""

from __future__ import annotations

import json
import logging
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional, Union

import librosa
import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio.transforms as T

# Add workspace to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from model.language_model import MultilingualLanguageModel

logger = logging.getLogger(__name__)


class Conv2dSubsampling(nn.Module):
    def __init__(self, input_dim: int = 80, hidden_dim: int = 256, conv_channels: int = 256) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, conv_channels, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(conv_channels, conv_channels, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
        )
        self.out_proj = nn.Linear(conv_channels * (input_dim // 4), hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(1)
        x = self.conv(x)
        b, c, f, t = x.size()
        x = x.permute(0, 3, 1, 2).contiguous().view(b, t, c * f)
        return self.out_proj(x)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1) -> None:
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


class ProprietarySTTModel(nn.Module):
    def __init__(
        self,
        input_dim: int = 80,
        hidden_dim: int = 256,
        num_layers: int = 6,
        num_heads: int = 4,
        vocab_size: int = 196,
        dropout: float = 0.1,
        blank_id: int = 2,
    ) -> None:
        super().__init__()
        self.blank_id = blank_id
        self.subsampling = Conv2dSubsampling(input_dim, hidden_dim)
        self.pos_encoder = PositionalEncoding(hidden_dim, dropout=dropout)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.ctc_linear = nn.Linear(hidden_dim, vocab_size)

    def forward(self, features: torch.Tensor, feature_lengths: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.subsampling(features)
        out_lengths = (feature_lengths // 4).clamp(min=1)
        x = self.pos_encoder(x)
        max_len = x.size(1)
        mask = torch.arange(max_len, device=x.device).unsqueeze(0) >= out_lengths.unsqueeze(1)
        encoded = self.layer_norm(self.transformer(x, src_key_padding_mask=mask))
        logits = self.ctc_linear(encoded)
        return F.log_softmax(logits, dim=-1), out_lengths


class ProprietarySTTInference:
    """Production inference engine for Proprietary Multilingual STT Model."""

    def __init__(
        self,
        model_path: str = "models/proprietary/proprietary_stt_model.pt",
        vocab_path: str = "models/proprietary/vocab.json",
        lm_path: Optional[str] = "models/proprietary/lm_data.json",
        device: str = "auto",
    ) -> None:
        if device == "auto":
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            elif torch.backends.mps.is_available():
                self.device = torch.device("mps")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)

        with open(vocab_path, "r", encoding="utf-8") as f:
            self.vocab = json.load(f)
        self.id2char = {idx: token for token, idx in self.vocab.items()}

        logger.info(f"Loading Proprietary STT model from {model_path} onto {self.device}...")
        ckpt = torch.load(model_path, map_location=self.device)
        self.model = ProprietarySTTModel(vocab_size=len(self.vocab))
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.to(self.device)
        self.model.eval()

        self.mel_transform = T.MelSpectrogram(
            sample_rate=16000, n_fft=400, win_length=400, hop_length=160, n_mels=80
        )

        self.lm: Optional[MultilingualLanguageModel] = None
        if lm_path and os.path.isfile(lm_path):
            self.lm = MultilingualLanguageModel(lm_path)
            logger.info("Multilingual Language Model loaded successfully.")

    def transcribe(
        self,
        audio_input: Union[str, Path, np.ndarray],
        sample_rate: int = 16000,
        use_lm: bool = True,
    ) -> Dict[str, Any]:
        """Transcribe an audio file or raw waveform numpy array."""
        t0 = time.time()

        if isinstance(audio_input, (str, Path)):
            audio_path = str(audio_input)
            try:
                audio, sr = sf.read(audio_path)
                if audio.ndim > 1:
                    audio = audio.mean(axis=1)
                if sr != sample_rate:
                    audio = librosa.resample(audio, orig_sr=sr, target_sr=sample_rate)
            except Exception:
                audio, _ = librosa.load(audio_path, sr=sample_rate)
            file_name = os.path.basename(audio_path)
        else:
            audio = np.asarray(audio_input, dtype=np.float32)
            if audio.ndim > 1:
                audio = audio.mean(axis=1)
            file_name = "live_audio_buffer"

        duration = len(audio) / sample_rate
        waveform = torch.from_numpy(audio).float()
        if waveform.abs().max() > 0:
            waveform = (waveform / waveform.abs().max()) * 0.95

        mel = self.mel_transform(waveform)
        log_mel = torch.log(torch.clamp(mel, min=1e-5))
        norm_mel = (log_mel - log_mel.mean(dim=-1, keepdim=True)) / (log_mel.std(dim=-1, keepdim=True) + 1e-5)

        features = norm_mel.unsqueeze(0).to(self.device)
        feature_lengths = torch.tensor([norm_mel.size(-1)], device=self.device)

        with torch.no_grad():
            log_probs, _ = self.model(features, feature_lengths)
            pred_ids = log_probs[0].argmax(dim=-1).cpu().tolist()

        # CTC collapse
        collapsed = []
        prev = None
        for tid in pred_ids:
            if tid in (0, 2):  # pad, blank
                prev = tid
                continue
            if tid != prev:
                collapsed.append(tid)
                prev = tid

        raw_text = "".join([self.id2char.get(i, "") for i in collapsed]).replace("|", " ").strip()

        # Language Model correction
        if use_lm and self.lm is not None and raw_text:
            final_text = self.lm.correct_transcript(raw_text)
        else:
            final_text = raw_text

        latency_ms = round((time.time() - t0) * 1000, 1)

        return {
            "file": file_name,
            "duration": round(duration, 2),
            "raw_text": raw_text,
            "text": final_text,
            "latency_ms": latency_ms,
            "lm_applied": use_lm and self.lm is not None,
        }
