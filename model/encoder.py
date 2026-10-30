"""
Acoustic Encoder — Converts Mel spectrograms into high-level acoustic representations.

Architecture:
    1. Convolutional Subsampling front-end (4x time downsampling).
    2. Sinusoidal Positional Encoding.
    3. Multi-layer Transformer Encoder with multi-head self-attention.
"""

from __future__ import annotations

import math
from typing import Tuple

import torch
import torch.nn as nn


class Conv2dSubsampling(nn.Module):
    """Convolutional 4x Subsampling layer for acoustic features.

    Converts (batch, 1, n_mels, time) -> (batch, time // 4, hidden_dim).
    """

    def __init__(self, input_dim: int = 80, hidden_dim: int = 256, conv_channels: int = 256) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, conv_channels, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(conv_channels, conv_channels, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
        )
        # After two stride-2 convs, frequency dimension is input_dim // 4
        downsampled_freq = input_dim // 4
        self.out_proj = nn.Linear(conv_channels * downsampled_freq, hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (batch, n_mels, time)

        Returns:
            Tensor of shape (batch, time // 4, hidden_dim)
        """
        # (batch, 1, n_mels, time)
        x = x.unsqueeze(1)
        x = self.conv(x)  # (batch, conv_channels, n_mels//4, time//4)
        b, c, f, t = x.size()
        x = x.permute(0, 3, 1, 2).contiguous().view(b, t, c * f)  # (batch, time//4, c*f)
        return self.out_proj(x)


class PositionalEncoding(nn.Module):
    """Sinusoidal Positional Encoding for sequence order information."""

    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1) -> None:
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (batch, seq_len, d_model)
        """
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


class Encoder(nn.Module):
    """Acoustic Encoder for Proprietary Speech-to-Text Model.

    Combines:
        1. 2D-CNN subsampling (4x reduction in sequence length)
        2. Sinusoidal positional embeddings
        3. Stack of Transformer Encoder layers
    """

    def __init__(
        self,
        input_dim: int = 80,
        hidden_dim: int = 256,
        num_layers: int = 6,
        num_heads: int = 4,
        dropout: float = 0.1,
        conv_channels: int = 256,
        conv_kernel_size: int = 3,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # Convolutional Front-End (4x time downsampling)
        self.subsampling = Conv2dSubsampling(input_dim=input_dim, hidden_dim=hidden_dim, conv_channels=conv_channels)

        # Positional Encoding
        self.pos_encoder = PositionalEncoding(d_model=hidden_dim, dropout=dropout)

        # Transformer Encoder Stack
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.layer_norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        features: torch.Tensor,
        feature_lengths: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode spectrogram features.

        Args:
            features: Tensor of shape (batch, n_mels, time)
            feature_lengths: Tensor of shape (batch,) containing audio lengths

        Returns:
            Tuple of:
                - encoded: Tensor of shape (batch, time // 4, hidden_dim)
                - out_lengths: Tensor of shape (batch,) containing updated lengths
        """
        # Subsampling (4x downsampling)
        x = self.subsampling(features)  # (batch, time//4, hidden_dim)

        # Updated lengths after 4x downsampling
        out_lengths = (feature_lengths // 4).clamp(min=1)

        # Positional Encoding
        x = self.pos_encoder(x)

        # Key padding mask: True indicates positions to be ignored
        max_len = x.size(1)
        padding_mask = torch.arange(max_len, device=x.device).unsqueeze(0) >= out_lengths.unsqueeze(1)

        # Transformer forward
        encoded = self.transformer_encoder(x, src_key_padding_mask=padding_mask)
        encoded = self.layer_norm(encoded)

        return encoded, out_lengths
