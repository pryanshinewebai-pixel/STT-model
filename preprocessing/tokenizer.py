"""
Character-Level CTC Tokenizer — Multilingual (English, Gujarati, Hindi).

Provides character-level tokenization suitable for CTC-based Speech-to-Text models
(e.g., Wav2Vec2, Conformer CTC, DeepSpeech).

Responsibility: Person 1 (Data Collection + Cleaning + Preprocessing)
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Dict, List, Union

logger = logging.getLogger(__name__)


class CTCTokenizer:
    """Character-level CTC tokenizer for multilingual text (English, Gujarati, Hindi).

    Special Tokens:
        <pad>: 0 (Padding token)
        <unk>: 1 (Unknown character token)
        <blank>: 2 (CTC blank token for CTCLoss)
        |: 3 (Word separator / space token)
    """

    PAD_TOKEN = "<pad>"
    UNK_TOKEN = "<unk>"
    BLANK_TOKEN = "<blank>"
    SPACE_TOKEN = "|"

    SPECIAL_TOKENS = [PAD_TOKEN, UNK_TOKEN, BLANK_TOKEN, SPACE_TOKEN]

    def __init__(self, vocab: Dict[str, int] | None = None) -> None:
        if vocab:
            self.char2id = dict(vocab)
        else:
            # Default initial special tokens
            self.char2id = {token: idx for idx, token in enumerate(self.SPECIAL_TOKENS)}

        self.id2char = {idx: token for token, idx in self.char2id.items()}

    def build_vocab(self, texts: List[str]) -> Dict[str, int]:
        """Build character vocabulary from a list of text transcripts."""
        unique_chars = set()

        for text in texts:
            cleaned = self.clean_text(text)
            for char in cleaned:
                if char != self.SPACE_TOKEN:
                    unique_chars.add(char)

        # Sort unique characters for deterministic vocabulary generation
        sorted_chars = sorted(list(unique_chars))

        # Re-initialize vocab starting with special tokens
        self.char2id = {token: idx for idx, token in enumerate(self.SPECIAL_TOKENS)}

        for char in sorted_chars:
            if char not in self.char2id:
                self.char2id[char] = len(self.char2id)

        self.id2char = {idx: token for token, idx in self.char2id.items()}
        logger.info(f"Built vocabulary with {len(self.char2id)} tokens.")
        return self.char2id

    def clean_text(self, text: str) -> str:
        """Clean and normalize transcript text for CTC tokenization."""
        if not text or text.startswith("["):
            return ""

        text = text.lower().strip()
        # Remove unwanted punctuation keeping Indic characters and spaces
        text = re.sub(r"[^\w\s\u0A80-\u0AFF\u0900-\u097F]", "", text)
        # Replace spaces with CTC space token '|'
        text = re.sub(r"\s+", self.SPACE_TOKEN, text)
        return text

    def encode(self, text: str) -> List[int]:
        """Convert a text transcript string into a list of CTC character token IDs."""
        cleaned_text = self.clean_text(text)
        token_ids = []

        for char in cleaned_text:
            token_id = self.char2id.get(char, self.char2id[self.UNK_TOKEN])
            token_ids.append(token_id)

        return token_ids

    def decode(self, token_ids: List[int], collapse_ctc: bool = False) -> str:
        """Convert a list of token IDs back into readable text string.

        Args:
            token_ids: List of integer token IDs.
            collapse_ctc: If True, collapses consecutive identical tokens and removes <blank>/<pad>.

        Returns:
            Decoded text string.
        """
        if collapse_ctc:
            # Standard CTC collapse algorithm
            collapsed = []
            prev_token = None
            blank_id = self.char2id.get(self.BLANK_TOKEN, 2)
            pad_id = self.char2id.get(self.PAD_TOKEN, 0)

            for tid in token_ids:
                if tid in (blank_id, pad_id):
                    prev_token = tid
                    continue
                if tid != prev_token:
                    collapsed.append(tid)
                    prev_token = tid

            chars = [self.id2char.get(tid, "") for tid in collapsed]
        else:
            chars = [self.id2char.get(tid, "") for tid in token_ids if tid not in (0, 1, 2)]

        text = "".join(chars).replace(self.SPACE_TOKEN, " ").strip()
        return text

    def save_vocab(self, json_path: str | Path) -> None:
        """Save vocabulary to a JSON file."""
        path = Path(json_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.char2id, f, ensure_ascii=False, indent=2)
        logger.info(f"Saved vocabulary to {path}")

    @classmethod
    def load_vocab(cls, json_path: str | Path) -> CTCTokenizer:
        """Load vocabulary from a JSON file and return a CTCTokenizer instance."""
        path = Path(json_path)
        if not path.is_file():
            raise FileNotFoundError(f"Vocab file not found: {path}")

        with open(path, "r", encoding="utf-8") as f:
            vocab = json.load(f)

        return cls(vocab=vocab)

    def __len__(self) -> int:
        return len(self.char2id)
