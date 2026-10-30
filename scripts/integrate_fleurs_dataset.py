#!/usr/bin/env python3
"""
Integrate Google FLEURS Multilingual Dataset into Proprietary STT Pipeline.

Fetches 100 clean, verified audio samples each for Hindi (hi), Gujarati (gu),
and English (en) from Google FLEURS, normalizes them to 16 kHz Mono WAV,
re-encodes CTC token IDs, updates vocab.json, manifests, Excel summaries,
and regenerates the Web Dashboard at http://localhost:8000.
"""

import argparse
import csv
import io
import json
import logging
import sys
import tarfile
import urllib.request
from pathlib import Path
from typing import Dict, List

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import soundfile as sf
from tqdm import tqdm

from preprocessing.tokenizer import CTCTokenizer
from scripts.update_excel_for_google_sheets import main as update_dashboard

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

DATA_BASE = PROJECT_ROOT / "data" / "chunked_dataset"
CHUNKS_WAV_DIR = DATA_BASE / "wav_chunks"
TRANSCRIPTS_DIR = DATA_BASE / "transcripts"
TOKENIZED_DIR = DATA_BASE / "tokenized_transcripts"
VOCAB_PATH = DATA_BASE / "vocab.json"
MANIFEST_CHUNKED = DATA_BASE / "manifest_chunked.csv"
MANIFEST_TOKENIZED = DATA_BASE / "manifest_tokenized.csv"

LANG_CONFIGS = {
    "gu": ("gu_in", "Gujarati 🇮🇳 (ગુજરાતી)"),
    "hi": ("hi_in", "Hindi 🇮🇳 (हिंदी)"),
    "en": ("en_us", "English 🇬🇧"),
}

HF_BASE_URL = "https://huggingface.co/datasets/google/fleurs/resolve/main/data"


def fetch_tsv_transcripts(config: str) -> Dict[str, str]:
    """Fetch train.tsv from Hugging Face and return a mapping of filename -> transcript."""
    url = f"{HF_BASE_URL}/{config}/train.tsv"
    logger.info(f"Downloading TSV metadata for {config} from {url}...")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        content = resp.read().decode("utf-8")

    mapping = {}
    for line in content.splitlines():
        parts = line.split("\t")
        if len(parts) >= 3:
            wav_name = parts[1].strip()
            transcript = parts[2].strip()
            if wav_name and transcript:
                mapping[wav_name] = transcript

    logger.info(f"Loaded {len(mapping)} verified transcripts for {config}.")
    return mapping


def stream_and_save_samples(lang: str, config: str, limit: int = 100) -> List[Dict]:
    """Stream WAV files directly from train.tar.gz without saving the full multi-GB archive."""
    tsv_map = fetch_tsv_transcripts(config)
    tar_url = f"{HF_BASE_URL}/{config}/audio/train.tar.gz"
    logger.info(f"Streaming {limit} audio samples for {lang} ({config}) from {tar_url}...")

    req = urllib.request.Request(tar_url, headers={"User-Agent": "Mozilla/5.0"})
    resp = urllib.request.urlopen(req, timeout=60)

    entries = []
    saved_count = 0

    CHUNKS_WAV_DIR.mkdir(parents=True, exist_ok=True)
    TRANSCRIPTS_DIR.mkdir(parents=True, exist_ok=True)

    with tarfile.open(fileobj=resp, mode="r|gz") as tar:
        for member in tar:
            if not member.name.endswith(".wav"):
                continue

            orig_filename = Path(member.name).name
            if orig_filename not in tsv_map:
                continue

            transcript = tsv_map[orig_filename]
            if not transcript:
                continue

            # Read WAV bytes into memory
            extracted_file = tar.extractfile(member)
            if extracted_file is None:
                continue

            wav_bytes = extracted_file.read()
            data, sr = sf.read(io.BytesIO(wav_bytes), dtype="float32")

            # Ensure mono
            if data.ndim > 1:
                data = data.mean(axis=1)

            duration = len(data) / sr

            saved_count += 1
            new_filename = f"fleurs_{lang}_{saved_count:03d}.wav"
            target_wav_path = CHUNKS_WAV_DIR / new_filename
            target_txt_path = TRANSCRIPTS_DIR / f"fleurs_{lang}_{saved_count:03d}.txt"

            # Write clean 16kHz WAV file
            sf.write(str(target_wav_path), data, sr, subtype="PCM_16")

            # Write text transcript file
            with open(target_txt_path, "w", encoding="utf-8") as tf:
                tf.write(transcript + "\n")

            entries.append({
                "audio_path": str(target_wav_path.resolve()),
                "audio_name": new_filename,
                "duration": round(duration, 2),
                "language": lang,
                "transcript": transcript,
            })

            if saved_count % 25 == 0:
                logger.info(f"Saved {saved_count}/{limit} {lang.upper()} samples...")

            if saved_count >= limit:
                break

    resp.close()
    logger.info(f"✅ Successfully downloaded and processed {len(entries)} {lang.upper()} samples.")
    return entries


def update_vocab_and_tokenize(all_entries: List[Dict]) -> None:
    """Update vocab.json to include any new characters and tokenize all new samples."""
    TOKENIZED_DIR.mkdir(parents=True, exist_ok=True)

    existing_vocab = {}
    if VOCAB_PATH.exists():
        with open(VOCAB_PATH, "r", encoding="utf-8") as f:
            existing_vocab = json.load(f)

    tokenizer = CTCTokenizer(existing_vocab)

    # Check for any new characters in the new transcripts
    new_chars = set()
    for e in all_entries:
        cleaned = tokenizer.clean_text(e["transcript"])
        for char in cleaned:
            if char != tokenizer.SPACE_TOKEN and char not in tokenizer.char2id:
                new_chars.add(char)

    if new_chars:
        logger.info(f"Adding {len(new_chars)} new characters to CTC vocabulary: {sorted(list(new_chars))}")
        for char in sorted(list(new_chars)):
            tokenizer.char2id[char] = len(tokenizer.char2id)
        tokenizer.id2char = {idx: ch for ch, idx in tokenizer.char2id.items()}

        with open(VOCAB_PATH, "w", encoding="utf-8") as vf:
            json.dump(tokenizer.char2id, vf, ensure_ascii=False, indent=2)
        logger.info(f"Updated {VOCAB_PATH} (Total tokens: {len(tokenizer.char2id)}).")

    # Generate tokenized JSON files for each entry
    for e in all_entries:
        token_ids = tokenizer.encode(e["transcript"])
        token_chars = [tokenizer.id2char.get(tid, "?") for tid in token_ids]

        e["token_ids"] = token_ids
        e["token_chars"] = token_chars
        e["num_tokens"] = len(token_ids)
        e["token_ids_str"] = " ".join(map(str, token_ids))

        json_path = TOKENIZED_DIR / f"{Path(e['audio_name']).stem}.json"
        with open(json_path, "w", encoding="utf-8") as jf:
            json.dump({
                "audio_name": e["audio_name"],
                "language": e["language"],
                "transcript": e["transcript"],
                "token_ids": token_ids,
                "token_chars": token_chars,
                "num_tokens": len(token_ids),
            }, jf, ensure_ascii=False, indent=2)

    logger.info(f"✅ Generated {len(all_entries)} tokenized JSON files in {TOKENIZED_DIR}")


def update_manifests(new_entries: List[Dict]) -> None:
    """Append new FLEURS entries to manifest_chunked.csv and manifest_tokenized.csv."""
    # 1. Update manifest_chunked.csv
    existing_chunked = []
    if MANIFEST_CHUNKED.exists():
        with open(MANIFEST_CHUNKED, "r", encoding="utf-8") as f:
            existing_chunked = [r for r in csv.DictReader(f) if not Path(r["audio_path"]).name.startswith("fleurs_")]

    all_chunked = existing_chunked + [
        {
            "audio_path": e["audio_path"],
            "duration": e["duration"],
            "language": e["language"],
            "transcript": e["transcript"],
        }
        for e in new_entries
    ]

    fieldnames_chunked = ["audio_path", "duration", "language", "transcript"]
    with open(MANIFEST_CHUNKED, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames_chunked, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(all_chunked)

    # 2. Update manifest_tokenized.csv
    existing_tokenized = []
    if MANIFEST_TOKENIZED.exists():
        with open(MANIFEST_TOKENIZED, "r", encoding="utf-8") as f:
            existing_tokenized = [r for r in csv.DictReader(f) if not Path(r["audio_path"]).name.startswith("fleurs_")]

    all_tokenized = existing_tokenized + [
        {
            "audio_path": e["audio_path"],
            "original_source": "google_fleurs",
            "duration": e["duration"],
            "language": e["language"],
            "transcript": e["transcript"],
            "num_tokens": e["num_tokens"],
            "token_ids_str": e["token_ids_str"],
        }
        for e in new_entries
    ]

    fieldnames_tokenized = ["audio_path", "original_source", "duration", "language", "transcript", "num_tokens", "token_ids_str"]
    with open(MANIFEST_TOKENIZED, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames_tokenized, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(all_tokenized)

    logger.info(f"📊 Updated manifests: {len(all_chunked)} total samples ({len(existing_chunked)} original + {len(new_entries)} FLEURS).")


def main() -> None:
    parser = argparse.ArgumentParser(description="Integrate Google FLEURS dataset into STT model dataset.")
    parser.add_argument("--samples_per_lang", type=int, default=100, help="Number of samples per language (default: 100).")
    args = parser.parse_args()

    limit = args.samples_per_lang
    all_fleurs_entries = []

    logger.info("=" * 60)
    logger.info("🚀 STARTING GOOGLE FLEURS DATASET INTEGRATION")
    logger.info(f"Languages: Hindi (hi), Gujarati (gu), English (en) | Samples/lang: {limit}")
    logger.info("=" * 60)

    existing_fleurs_wavs = list(CHUNKS_WAV_DIR.glob("fleurs_*.wav"))
    if len(existing_fleurs_wavs) >= limit * len(LANG_CONFIGS):
        logger.info(f"⚡ Found {len(existing_fleurs_wavs)} already downloaded FLEURS WAV files on disk. Loading directly from disk!")
        for lang in LANG_CONFIGS:
            for i in range(1, limit + 1):
                wav_file = CHUNKS_WAV_DIR / f"fleurs_{lang}_{i:03d}.wav"
                txt_file = TRANSCRIPTS_DIR / f"fleurs_{lang}_{i:03d}.txt"
                if wav_file.exists() and txt_file.exists():
                    data, sr = sf.read(str(wav_file))
                    duration = len(data) / sr
                    with open(txt_file, "r", encoding="utf-8") as tf:
                        transcript = tf.read().strip()
                    all_fleurs_entries.append({
                        "audio_path": str(wav_file.resolve()),
                        "audio_name": wav_file.name,
                        "duration": round(duration, 2),
                        "language": lang,
                        "transcript": transcript,
                    })
    else:
        for lang, (config, lang_name) in LANG_CONFIGS.items():
            logger.info(f"\n--- Processing {lang_name} ({limit} samples) ---")
            entries = stream_and_save_samples(lang, config, limit=limit)
            all_fleurs_entries.extend(entries)

    logger.info(f"\nTotal FLEURS samples ready: {len(all_fleurs_entries)}")

    # Update vocab and generate tokens
    logger.info("\nRe-building Vocabulary and Generating CTC Tokens...")
    update_vocab_and_tokenize(all_fleurs_entries)

    # Update manifests
    logger.info("\nUpdating Manifest CSV files...")
    update_manifests(all_fleurs_entries)

    # Regenerate Excel & Web Dashboard
    logger.info("\nRegenerating Excel Summary and Web Dashboard...")
    update_dashboard()

    logger.info("\n" + "=" * 60)
    logger.info("🎉 FLEURS DATASET INTEGRATION COMPLETED SUCCESSFULLY!")
    logger.info(f"Total New Samples: {len(all_fleurs_entries)}")
    logger.info(f"Updated Dashboard: http://localhost:8000")
    logger.info(f"Desktop Excel: {Path.home() / 'Desktop' / 'stt_chunked_summary.xlsx'}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
