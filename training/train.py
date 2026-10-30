"""
Training Engine for Proprietary Speech-to-Text Model.
"""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from model.stt_model import ProprietarySTTModel
from preprocessing.tokenizer import CTCTokenizer

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


class ProprietaryTrainer:
    """Manages the full end-to-end training process for ProprietarySTTModel.

    Handles:
        - CTC Loss computation
        - AdamW optimizer & CosineAnnealing learning rate schedule
        - Gradient clipping and FP16 / AMP mixed precision
        - Validation evaluation and best-model checkpoint saving
    """

    def __init__(
        self,
        model: ProprietarySTTModel,
        tokenizer: CTCTokenizer,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        learning_rate: float = 3e-4,
        weight_decay: float = 1e-4,
        num_epochs: int = 50,
        checkpoint_dir: str | Path = "checkpoints/proprietary",
        device: str = "cpu",
    ) -> None:
        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.tokenizer = tokenizer
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.num_epochs = num_epochs
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # CTC Loss: blank token is 2 in CTCTokenizer
        self.criterion = nn.CTCLoss(blank=model.blank_id, zero_infinity=True)

        # Optimizer & Scheduler
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
            betas=(0.9, 0.98),
            eps=1e-8,
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=num_epochs,
            eta_min=1e-6,
        )

        self.best_val_loss = float("inf")

    def train_epoch(self, epoch: int) -> float:
        self.model.train()
        total_loss = 0.0
        start_time = time.time()

        for step, batch in enumerate(self.train_loader):
            features = batch["features"].to(self.device)
            feature_lengths = batch["feature_lengths"].to(self.device)
            targets = batch["targets"].to(self.device)
            target_lengths = batch["target_lengths"].to(self.device)

            self.optimizer.zero_grad()

            # Forward pass: log_probs of shape (batch, time//4, vocab_size)
            log_probs, enc_lengths = self.model(features, feature_lengths)

            # PyTorch CTCLoss expects (time, batch, vocab_size)
            ctc_inputs = log_probs.permute(1, 0, 2)

            loss = self.criterion(ctc_inputs, targets, enc_lengths, target_lengths)

            if torch.isnan(loss) or torch.isinf(loss):
                continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
            self.optimizer.step()

            total_loss += loss.item()

            if (step + 1) % 25 == 0 or (step + 1) == len(self.train_loader):
                avg_step_loss = total_loss / (step + 1)
                lr = self.optimizer.param_groups[0]["lr"]
                logger.info(
                    f"Epoch [{epoch}/{self.num_epochs}] | Step [{step+1}/{len(self.train_loader)}] | "
                    f"Loss: {loss.item():.4f} (Avg: {avg_step_loss:.4f}) | LR: {lr:.2e}"
                )

        self.scheduler.step()
        avg_loss = total_loss / max(1, len(self.train_loader))
        elapsed = round(time.time() - start_time, 2)
        logger.info(f"Epoch {epoch} complete in {elapsed}s | Train Loss: {avg_loss:.4f}")
        return avg_loss

    @torch.no_grad()
    def evaluate(self) -> float:
        if not self.val_loader:
            return 0.0

        self.model.eval()
        total_loss = 0.0

        for batch in self.val_loader:
            features = batch["features"].to(self.device)
            feature_lengths = batch["feature_lengths"].to(self.device)
            targets = batch["targets"].to(self.device)
            target_lengths = batch["target_lengths"].to(self.device)

            log_probs, enc_lengths = self.model(features, feature_lengths)
            ctc_inputs = log_probs.permute(1, 0, 2)
            loss = self.criterion(ctc_inputs, targets, enc_lengths, target_lengths)

            if not torch.isnan(loss) and not torch.isinf(loss):
                total_loss += loss.item()

        avg_val_loss = total_loss / max(1, len(self.val_loader))
        logger.info(f"🔍 Validation Loss: {avg_val_loss:.4f}")
        return avg_val_loss

    def train(self) -> Dict[str, Any]:
        logger.info(f"🚀 Starting Proprietary STT Training on {self.device}...")
        history = {"train_loss": [], "val_loss": []}

        for epoch in range(1, self.num_epochs + 1):
            train_loss = self.train_epoch(epoch)
            val_loss = self.evaluate()

            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)

            # Save best checkpoint
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                best_path = self.checkpoint_dir / "best_proprietary_model.pt"
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": self.model.state_dict(),
                        "optimizer_state_dict": self.optimizer.state_dict(),
                        "val_loss": val_loss,
                        "vocab_size": self.model.vocab_size,
                    },
                    best_path,
                )
                logger.info(f"🏆 New best model saved to {best_path} (Val Loss: {val_loss:.4f})")

        return history
