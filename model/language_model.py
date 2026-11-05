"""
Multilingual N-Gram Language Model and CTC Beam Search Decoder
Supports Gujarati, Hindi, and English vocabulary with n-gram contextual scoring
and phonetic / minimum edit distance correction.
"""

from __future__ import annotations

import json
import math
import os
import re
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np


class MultilingualLanguageModel:
    """N-Gram Language Model (Unigram + Bigram) with Levenshtein-based

    lexicon correction and CTC Prefix Beam Search for Gujarati, Hindi, and English.
    """

    def __init__(self, lm_path: Optional[str] = "models/proprietary/lm_data.json") -> None:
        self.unigrams: Dict[str, int] = {}
        self.bigrams: Dict[str, Dict[str, int]] = {}
        self.total_words: int = 1
        self.vocab_size: int = 1
        self.word_list: List[str] = []
        self._words_by_len: Dict[int, List[str]] = defaultdict(list)

        if lm_path and os.path.isfile(lm_path):
            self.load(lm_path)

    def load(self, lm_path: str) -> None:
        """Load trained language model statistics from a JSON file."""
        with open(lm_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.total_words = data.get("total_words", 1)
        self.unigrams = data.get("unigrams", {})
        self.bigrams = data.get("bigrams", {})
        self.vocab_size = len(self.unigrams)
        self.word_list = list(self.unigrams.keys())

        # Index words by length for fast candidate retrieval
        self._words_by_len = defaultdict(list)
        for w in self.word_list:
            self._words_by_len[len(w)].append(w)

    def score_unigram(self, word: str) -> float:
        """Calculate log probability of a single unigram word with Laplace smoothing."""
        count = self.unigrams.get(word, 0)
        return math.log((count + 1) / (self.total_words + self.vocab_size))

    def score_bigram(self, prev_word: Optional[str], current_word: str) -> float:
        """Calculate log probability of word given previous word with Lidstone smoothing."""
        if not prev_word or prev_word not in self.bigrams:
            return self.score_unigram(current_word)

        bi_dict = self.bigrams[prev_word]
        count_bi = bi_dict.get(current_word, 0)
        count_prev = self.unigrams.get(prev_word, 0)
        if count_prev == 0:
            return self.score_unigram(current_word)

        # Interpolate bigram with unigram
        p_bi = (count_bi + 0.1) / (count_prev + 0.1 * self.vocab_size)
        return math.log(max(1e-9, p_bi))

    @staticmethod
    def levenshtein(s1: str, s2: str) -> int:
        """Compute standard Levenshtein edit distance between two strings."""
        if len(s1) < len(s2):
            return MultilingualLanguageModel.levenshtein(s2, s1)
        if len(s2) == 0:
            return len(s1)
        previous_row = range(len(s2) + 1)
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row
        return previous_row[-1]

    def correct_word(
        self,
        raw_word: str,
        prev_word: Optional[str] = None,
        max_dist: int = 2,
        dist_penalty: float = 3.5,
    ) -> str:
        """Find the most probable lexicon word for a raw noisy word."""
        if not raw_word or len(raw_word) < 2:
            return raw_word

        # If it's already an exact known word with sufficient frequency
        if raw_word in self.unigrams and self.unigrams[raw_word] >= 2:
            return raw_word

        raw_len = len(raw_word)
        candidates = []
        for l in range(max(1, raw_len - max_dist), raw_len + max_dist + 1):
            candidates.extend(self._words_by_len.get(l, []))

        if not candidates:
            return raw_word

        best_word = raw_word
        best_score = -float("inf")

        for cand in candidates:
            # Quick first letter check if long enough
            if raw_len > 4 and cand[0] != raw_word[0]:
                continue

            dist = self.levenshtein(raw_word, cand)
            if dist <= max_dist:
                log_p_uni = self.score_unigram(cand)
                log_p_bi = self.score_bigram(prev_word, cand)
                score = -dist * dist_penalty + 0.7 * log_p_uni + 0.3 * log_p_bi

                if score > best_score:
                    best_score = score
                    best_word = cand

        return best_word

    def correct_transcript(self, transcript: str) -> str:
        """Clean and correct a full sentence transcript using n-gram language model."""
        if not transcript:
            return ""

        words = transcript.strip().split()
        corrected_words: List[str] = []
        prev_word: Optional[str] = None

        for w in words:
            cw = self.correct_word(w, prev_word=prev_word)
            corrected_words.append(cw)
            prev_word = cw

        return " ".join(corrected_words)

    def ctc_prefix_beam_search(
        self,
        log_probs: np.ndarray,
        id2char: Dict[int, str],
        blank_id: int = 2,
        space_char: str = "|",
        beam_width: int = 10,
        lm_weight: float = 0.4,
    ) -> str:
        """CTC Prefix Beam Search decoder with LM score bonus on word endings."""
        T, V = log_probs.shape
        beam = {"": (1.0, 0.0)}  # prefix -> (P_blank, P_non_blank)

        for t in range(T):
            new_beam = defaultdict(lambda: [0.0, 0.0])
            top_k = np.argsort(log_probs[t])[-15:]

            for prefix, (p_b, p_nb) in beam.items():
                p_total = p_b + p_nb

                for c_id in top_k:
                    p_c = float(np.exp(log_probs[t, c_id]))

                    if c_id == blank_id:
                        new_beam[prefix][0] += p_total * p_c
                    else:
                        char = id2char.get(c_id, "")
                        if not char:
                            continue
                        new_prefix = prefix + char

                        if len(prefix) > 0 and prefix[-1] == char:
                            new_beam[prefix][1] += p_nb * p_c
                            new_beam[new_prefix][1] += p_b * p_c
                        else:
                            new_beam[new_prefix][1] += p_total * p_c

            def beam_score(item: Tuple[str, Tuple[float, float]]) -> float:
                pref, (pb, pnb) = item
                base_prob = pb + pnb
                if lm_weight > 0 and pref.endswith(space_char):
                    last_word = pref.rstrip(space_char).split(space_char)[-1]
                    lm_bonus = math.exp(self.score_unigram(last_word) * lm_weight)
                    base_prob *= lm_bonus
                return base_prob

            sorted_beam = sorted(new_beam.items(), key=beam_score, reverse=True)[:beam_width]
            beam = {k: (v[0], v[1]) for k, v in sorted_beam}

        if not beam:
            return ""

        best_prefix = max(beam.items(), key=lambda x: x[1][0] + x[1][1])[0]
        raw_text = best_prefix.replace(space_char, " ").strip()
        # Apply final language model polish
        return self.correct_transcript(raw_text)
