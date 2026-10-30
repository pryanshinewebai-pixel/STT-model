"""
Proprietary STT Model — End-to-End Multilingual Speech-to-Text Model.

Combines:
    - 2D-CNN Time Subsampling (4x downsampling)
    - Deep Transformer Acoustic Encoder
    - CTC Prediction Head with Greedy & Beam Search Decoding
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.encoder import Encoder


class ProprietarySTTModel(nn.Module):
    """End-to-End Proprietary Speech-to-Text Model for Indic & English speech.

    Args:
        input_dim: Number of input Mel filterbanks (default: 80).
        hidden_dim: Hidden dimension of encoder layers (default: 256).
        num_layers: Number of Transformer encoder layers (default: 6).
        num_heads: Number of attention heads (default: 4).
        vocab_size: Total vocabulary size (including special tokens).
        dropout: Dropout rate (default: 0.1).
        blank_id: Token ID for CTC blank (default: 2).
    """

    def __init__(
        self,
        input_dim: int = 80,
        hidden_dim: int = 256,
        num_layers: int = 6,
        num_heads: int = 4,
        vocab_size: int = 200,
        dropout: float = 0.1,
        blank_id: int = 2,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.blank_id = blank_id

        # Acoustic Encoder
        self.encoder = Encoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            dropout=dropout,
        )

        # CTC Output Projection Layer
        self.ctc_linear = nn.Linear(hidden_dim, vocab_size)

    def forward(
        self,
        features: torch.Tensor,
        feature_lengths: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute log-probabilities for CTC loss.

        Args:
            features: Mel spectrogram tensor of shape (batch, n_mels, time).
            feature_lengths: 1D tensor of original audio frame lengths.

        Returns:
            Tuple of:
                - log_probs: Tensor of shape (batch, time // 4, vocab_size) with log_softmax applied.
                - enc_lengths: Tensor of shape (batch,) with lengths after 4x downsampling.
        """
        # Encoder forward (time dimension is subsampled 4x)
        encoded, enc_lengths = self.encoder(features, feature_lengths)

        # Compute logits and log_softmax for CTCLoss
        logits = self.ctc_linear(encoded)  # (batch, time//4, vocab_size)
        log_probs = F.log_softmax(logits, dim=-1)

        return log_probs, enc_lengths

    @torch.no_grad()
    def transcribe(
        self,
        features: torch.Tensor,
        feature_lengths: torch.Tensor,
        tokenizer: Optional[Any] = None,
    ) -> List[str]:
        """Inference transcription using Greedy CTC Decoding.

        Args:
            features: Tensor of shape (batch, n_mels, time).
            feature_lengths: Tensor of shape (batch,).
            tokenizer: CTCTokenizer instance to decode token IDs to text.

        Returns:
            List of transcribed text strings.
        """
        self.eval()
        log_probs, enc_lengths = self.forward(features, feature_lengths)
        predictions = torch.argmax(log_probs, dim=-1).cpu().numpy()  # (batch, time)

        transcripts = []
        for b_idx in range(len(predictions)):
            raw_tokens = predictions[b_idx][: enc_lengths[b_idx].item()]

            # CTC Collapse algorithm: collapse consecutive duplicates and drop blank
            collapsed = []
            prev_token = None
            for token in raw_tokens:
                if token == self.blank_id:
                    prev_token = token
                    continue
                if token != prev_token:
                    collapsed.append(int(token))
                    prev_token = token

            if tokenizer is not None:
                text = tokenizer.decode(collapsed, collapse_ctc=False)
            else:
                text = " ".join(map(str, collapsed))

            transcripts.append(text)

        return transcripts

    def count_parameters(self) -> int:
        """Return total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# Backward compatibility alias
STTModel = ProprietarySTTModel
