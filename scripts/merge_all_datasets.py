#!/usr/bin/env python3
"""
Master Dataset Merging Pipeline.

Merges:
  1. 728 original audio chunks (English, Hindi, Gujarati)
  2. 300 Google FLEURS verified speech clips
  3. 565 recorded voice studio audio clips from Desktop/STT_model

Total: 1,593 unified audio samples with 100% verified ground truth transcripts.
Updates:
  - wav_chunks/ (audio clips)
  - transcripts/ (.txt files)
  - tokenized_transcripts/ (.json files)
  - vocab.json (multilingual CTC vocabulary)
  - manifest_chunked.csv & manifest_tokenized.csv
  - stt_chunked_summary.xlsx (project & Desktop)
  - index.html (Audio Dashboard at http://localhost:8000)
  - colab_stt_dataset.zip (updated training package on Desktop)
"""

import csv
import json
import logging
import shutil
import sys
import zipfile
from pathlib import Path
from typing import Dict, List

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import soundfile as sf

from preprocessing.tokenizer import CTCTokenizer
from scripts.update_excel_for_google_sheets import main as update_dashboard

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

DESKTOP_STT_DIR = Path.home() / "Desktop" / "STT_model"
DESKTOP_METADATA = DESKTOP_STT_DIR / "data" / "metadata" / "metadata.csv"
DESKTOP_RECORDINGS = DESKTOP_STT_DIR / "data" / "recordings"

DATA_BASE = PROJECT_ROOT / "data" / "chunked_dataset"
CHUNKS_WAV_DIR = DATA_BASE / "wav_chunks"
TRANSCRIPTS_DIR = DATA_BASE / "transcripts"
TOKENIZED_DIR = DATA_BASE / "tokenized_transcripts"
VOCAB_PATH = DATA_BASE / "vocab.json"
MANIFEST_CHUNKED = DATA_BASE / "manifest_chunked.csv"
MANIFEST_TOKENIZED = DATA_BASE / "manifest_tokenized.csv"


def merge_desktop_recordings() -> List[Dict]:
    """Load and copy 565 recordings from Desktop/STT_model into proprietary_stt_model."""
    if not DESKTOP_METADATA.exists():
        logger.warning(f"Desktop metadata not found at: {DESKTOP_METADATA}")
        return []

    with open(DESKTOP_METADATA, "r", encoding="utf-8") as f:
        reader = list(csv.DictReader(f))

    logger.info(f"Loaded {len(reader)} recordings from Desktop/STT_model metadata.")
    desktop_entries = []

    for r in reader:
        wav_name = r["file_name"]
        src_wav = DESKTOP_RECORDINGS / wav_name
        if not src_wav.exists():
            continue

        target_wav = CHUNKS_WAV_DIR / wav_name
        target_txt = TRANSCRIPTS_DIR / f"{Path(wav_name).stem}.txt"

        # Copy WAV if not present
        if not target_wav.exists():
            shutil.copy2(src_wav, target_wav)

        transcript = r["transcription"].strip()
        # Save text transcript
        with open(target_txt, "w", encoding="utf-8") as tf:
            tf.write(transcript + "\n")

        duration = float(r.get("duration_sec", 0.0))
        if duration == 0.0:
            data, sr = sf.read(str(src_wav))
            duration = round(len(data) / sr, 2)

        desktop_entries.append({
            "audio_path": str(target_wav.resolve()),
            "audio_name": wav_name,
            "original_source": r.get("source_file", "voice_studio"),
            "duration": round(duration, 2),
            "language": r["language"],
            "transcript": transcript,
        })

    logger.info(f"✅ Integrated {len(desktop_entries)} recordings from Desktop/STT_model.")
    return desktop_entries


def update_vocab_and_tokens(new_entries: List[Dict]) -> None:
    """Update vocab.json and create tokenized JSON files for all entries."""
    with open(VOCAB_PATH, "r", encoding="utf-8") as vf:
        vocab = json.load(vf)

    tokenizer = CTCTokenizer(vocab)

    # Check for new characters
    new_chars = set()
    for e in new_entries:
        cleaned = tokenizer.clean_text(e["transcript"])
        for char in cleaned:
            if char != tokenizer.SPACE_TOKEN and char not in tokenizer.char2id:
                new_chars.add(char)

    if new_chars:
        logger.info(f"Adding {len(new_chars)} new characters to vocab: {sorted(list(new_chars))}")
        for char in sorted(list(new_chars)):
            tokenizer.char2id[char] = len(tokenizer.char2id)
        tokenizer.id2char = {idx: ch for ch, idx in tokenizer.char2id.items()}
        with open(VOCAB_PATH, "w", encoding="utf-8") as vf:
            json.dump(tokenizer.char2id, vf, ensure_ascii=False, indent=2)

    # Generate tokens for each new entry
    for e in new_entries:
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


def sync_manifests(desktop_entries: List[Dict]) -> int:
    """Merge desktop entries with existing manifest_tokenized.csv and manifest_chunked.csv."""
    existing_tokenized = []
    if MANIFEST_TOKENIZED.exists():
        with open(MANIFEST_TOKENIZED, "r", encoding="utf-8") as f:
            existing_tokenized = [r for r in csv.DictReader(f) if not Path(r["audio_path"]).name.startswith("recording_prompt_")]

    all_tokenized = existing_tokenized + [
        {
            "audio_path": e["audio_path"],
            "original_source": e["original_source"],
            "duration": e["duration"],
            "language": e["language"],
            "transcript": e["transcript"],
            "num_tokens": e["num_tokens"],
            "token_ids_str": e["token_ids_str"],
        }
        for e in desktop_entries
    ]

    fieldnames_tok = ["audio_path", "original_source", "duration", "language", "transcript", "num_tokens", "token_ids_str"]
    with open(MANIFEST_TOKENIZED, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames_tok, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(all_tokenized)

    fieldnames_chunk = ["audio_path", "duration", "language", "transcript"]
    with open(MANIFEST_CHUNKED, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames_chunk, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(all_tokenized)

    total_count = len(all_tokenized)
    logger.info(f"📊 Manifests synced: {total_count} total audio samples ({len(existing_tokenized)} existing + {len(desktop_entries)} new).")
    return total_count


def package_colab_zip(total_count: int) -> None:
    """Package the entire unified dataset (all 1,590+ audio files) into colab_stt_dataset.zip."""
    zip_dest = PROJECT_ROOT / "colab_stt_dataset.zip"
    desktop_dest = Path.home() / "Desktop" / "colab_stt_dataset.zip"

    logger.info(f"📦 Packaging {total_count} audio files into {zip_dest.name}...")

    with open(MANIFEST_CHUNKED, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    with zipfile.ZipFile(zip_dest, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        csv_buf = "audio_path,file_name,duration,language,transcript\n"
        for r in rows:
            fname = Path(r["audio_path"]).name
            t = r["transcript"].replace('"', '""')
            csv_buf += f"wav_chunks/{fname},{fname},{r['duration']},{r['language']},\"{t}\"\n"
        zf.writestr("manifest.csv", csv_buf.encode("utf-8"))

        if VOCAB_PATH.exists():
            zf.write(VOCAB_PATH, arcname="vocab.json")

        for wf in sorted(list(CHUNKS_WAV_DIR.glob("*.wav"))):
            zf.write(wf, arcname=f"wav_chunks/{wf.name}")

    shutil.copy2(zip_dest, desktop_dest)
    size_mb = zip_dest.stat().st_size / (1024 * 1024)
    logger.info(f"✅ Updated colab_stt_dataset.zip ({size_mb:.2f} MB) on Desktop!")


def main() -> None:
    logger.info("=" * 65)
    logger.info("🚀 STARTING UNIFIED DATASET MERGE")
    logger.info("=" * 65)

    desktop_entries = merge_desktop_recordings()
    if desktop_entries:
        logger.info("\nUpdating Vocab and Generating CTC Tokens for Desktop recordings...")
        update_vocab_and_tokens(desktop_entries)

    logger.info("\nSyncing Manifest CSV Files...")
    total_count = sync_manifests(desktop_entries)

    logger.info("\nRegenerating Excel Summary and Web Audio Dashboard...")
    update_dashboard()

    logger.info("\nUpdating Master Training Package for Colab...")
    package_colab_zip(total_count)

    logger.info("\n" + "=" * 65)
    logger.info("🎉 MASTER DATASET MERGED SUCCESSFULLY!")
    logger.info(f"Total Unified Samples: {total_count}")
    logger.info(f"Audio Dashboard       : http://localhost:8000")
    logger.info(f"Live Web Tester       : http://localhost:8080")
    logger.info(f"Colab Dataset Zip     : {Path.home() / 'Desktop' / 'colab_stt_dataset.zip'}")
    logger.info("=" * 65)


if __name__ == "__main__":
    main()
