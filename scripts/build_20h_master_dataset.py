#!/usr/bin/env python3
"""
20+ Hours Multilingual Master STT Dataset Builder.

Expands dataset to 20+ Hours (~7,400 audio samples) across:
  - Gujarati (gu): ~6.8 hours (~2,370 samples)
  - Hindi (hi):    ~7.0 hours (~2,480 samples)
  - English (en):  ~6.8 hours (~2,540 samples)

Adds 3,000 new verified native speech samples (1,000 Gujarati, 1,000 Hindi, 1,000 English)
to the existing 4,393 samples with 100% deduplication.

Outputs:
  - data/chunked_dataset/wav_chunks/ (16 kHz Mono WAVs, peak normalized to 0.95)
  - data/chunked_dataset/transcripts/ (.txt native script files)
  - data/chunked_dataset/manifest_chunked.csv (Master manifest)
  - /Users/commerciax/Desktop/colab_stt_dataset.zip (~1.8 GB ready for Kaggle/Colab GPU)
"""

import csv
import io
import json
import logging
import os
import shutil
import sys
import tarfile
import zipfile
from pathlib import Path
from typing import Dict, List, Set

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import requests
import soundfile as sf
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("20H_Builder")

DATA_BASE = PROJECT_ROOT / "data" / "chunked_dataset"
CHUNKS_WAV_DIR = DATA_BASE / "wav_chunks"
TRANSCRIPTS_DIR = DATA_BASE / "transcripts"
MANIFEST_CHUNKED = DATA_BASE / "manifest_chunked.csv"
DESKTOP_ZIP_PATH = Path.home() / "Desktop" / "colab_stt_dataset.zip"
LOCAL_ZIP_PATH = PROJECT_ROOT / "colab_stt_dataset.zip"

LANG_TARGETS = {
    "gu": {"config": "gu_in", "target_new": 1000, "name": "Gujarati 🇮🇳 (ગુજરાતી)"},
    "hi": {"config": "hi_in", "target_new": 1000, "name": "Hindi 🇮🇳 (हिंदी)"},
    "en": {"config": "en_us", "target_new": 1000, "name": "English 🇬🇧"},
}

HF_BASE_URL = "https://huggingface.co/datasets/google/fleurs/resolve/main/data"


def fetch_tsv_transcripts(config: str, split: str = "train") -> Dict[str, str]:
    """Fetch TSV metadata for a split from Hugging Face and return a mapping of filename -> transcript."""
    url = f"{HF_BASE_URL}/{config}/{split}.tsv"
    logger.info(f"Downloading TSV metadata for {config} ({split}) from {url}...")
    headers = {"User-Agent": "Mozilla/5.0"}
    resp = requests.get(url, headers=headers, timeout=30)
    resp.raise_for_status()

    mapping = {}
    for line in resp.text.splitlines():
        parts = line.split("\t")
        if len(parts) >= 3:
            wav_name = parts[1].strip()
            transcript = parts[2].strip()
            if wav_name and transcript:
                mapping[wav_name] = transcript

    logger.info(f"Loaded {len(mapping)} verified transcripts for {config} ({split}).")
    return mapping


def stream_fleurs_split(
    lang: str,
    config: str,
    split: str,
    target_needed: int,
    existing_transcripts: Set[str],
    current_saved: int
) -> List[Dict]:
    """Stream WAV files directly from split tar.gz with deduplication."""
    tsv_map = fetch_tsv_transcripts(config, split)
    tar_url = f"{HF_BASE_URL}/{config}/audio/{split}.tar.gz"
    logger.info(f"Streaming {target_needed} samples for {lang.upper()} ({config} - {split}) from {tar_url}...")

    headers = {"User-Agent": "Mozilla/5.0"}
    resp = requests.get(tar_url, stream=True, headers=headers, timeout=60)
    resp.raise_for_status()

    entries = []
    saved_in_this_split = 0

    CHUNKS_WAV_DIR.mkdir(parents=True, exist_ok=True)
    TRANSCRIPTS_DIR.mkdir(parents=True, exist_ok=True)

    with tarfile.open(fileobj=resp.raw, mode="r|gz") as tar:
        for member in tar:
            if not member.name.endswith(".wav"):
                continue

            orig_filename = Path(member.name).name
            if orig_filename not in tsv_map:
                continue

            transcript = tsv_map[orig_filename].strip()
            if not transcript or len(transcript) < 2:
                continue

            # Strict Deduplication: skip if transcript was already in dataset
            if transcript in existing_transcripts:
                continue

            extracted_file = tar.extractfile(member)
            if extracted_file is None:
                continue

            wav_bytes = extracted_file.read()
            data, sr = sf.read(io.BytesIO(wav_bytes), dtype="float32")

            # Ensure mono
            if data.ndim > 1:
                data = data.mean(axis=1)

            # Peak normalize audio to 0.95
            max_amp = float(abs(data).max()) if len(data) > 0 else 0.0
            if max_amp > 1e-5:
                data = (data / max_amp) * 0.95

            duration = len(data) / sr

            current_saved += 1
            saved_in_this_split += 1
            new_filename = f"fleurs_20h_{lang}_{current_saved:04d}.wav"
            target_wav_path = CHUNKS_WAV_DIR / new_filename
            target_txt_path = TRANSCRIPTS_DIR / f"fleurs_20h_{lang}_{current_saved:04d}.txt"

            # Write clean 16kHz WAV file
            sf.write(str(target_wav_path), data, sr, subtype="PCM_16")

            # Write text transcript file
            with open(target_txt_path, "w", encoding="utf-8") as tf:
                tf.write(transcript + "\n")

            # Mark as existing so later iterations don't duplicate
            existing_transcripts.add(transcript)

            entries.append({
                "audio_path": str(target_wav_path.resolve()),
                "audio_name": new_filename,
                "duration": round(duration, 2),
                "language": lang,
                "transcript": transcript,
            })

            if saved_in_this_split % 100 == 0 or saved_in_this_split == target_needed:
                logger.info(f"Progress [{lang.upper()} - {split}]: {saved_in_this_split}/{target_needed} saved.")

            if saved_in_this_split >= target_needed:
                break

    resp.close()
    return entries


def create_training_zip(manifest_rows: List[Dict]):
    """Packages all WAV files and manifest.csv into a unified colab_stt_dataset.zip."""
    logger.info("📦 Packaging 20+ Hour Dataset into colab_stt_dataset.zip...")

    temp_manifest_path = DATA_BASE / "manifest.csv"
    with open(temp_manifest_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["audio_path", "duration", "language", "transcript"])
        writer.writeheader()
        for r in manifest_rows:
            wav_name = Path(r["audio_path"]).name
            writer.writerow({
                "audio_path": f"wav_chunks/{wav_name}",
                "duration": r["duration"],
                "language": r["language"],
                "transcript": r["transcript"],
            })

    total_files = len(manifest_rows)
    logger.info(f"Adding {total_files} audio files to zip...")

    for zip_out in [LOCAL_ZIP_PATH, DESKTOP_ZIP_PATH]:
        logger.info(f"Writing zip to: {zip_out}...")
        with zipfile.ZipFile(zip_out, "w", compression=zipfile.ZIP_DEFLATED) as z:
            z.write(temp_manifest_path, arcname="manifest.csv")

            for r in tqdm(manifest_rows, desc="Zipping Audio Files"):
                wav_path = Path(r["audio_path"])
                if wav_path.exists():
                    z.write(wav_path, arcname=f"wav_chunks/{wav_path.name}")

        size_mb = os.path.getsize(zip_out) / (1024 * 1024)
        logger.info(f"🎉 Created {zip_out} ({size_mb:.2f} MB)")


def main():
    logger.info("🚀 Starting 20+ Hours Master STT Dataset Expansion...")

    # 1. Read existing manifest
    existing_rows = []
    existing_transcripts = set()
    if MANIFEST_CHUNKED.exists():
        with open(MANIFEST_CHUNKED, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                existing_rows.append(row)
                t = row.get("transcript", "").strip()
                if t:
                    existing_transcripts.add(t)
        logger.info(f"Loaded {len(existing_rows)} existing samples ({len(existing_transcripts)} unique transcripts).")
    else:
        logger.error(f"Cannot find {MANIFEST_CHUNKED}!")
        return

    # 2. Ingest 1,000 new samples per language (gu, hi, en)
    all_new_entries = []
    for lang, info in LANG_TARGETS.items():
        logger.info(f"\n==========================================")
        logger.info(f"Expanding Language: {info['name']} (Target: +{info['target_new']} clips)")
        logger.info(f"==========================================")

        needed = info["target_new"]
        lang_entries = []

        # Try train split first
        train_entries = stream_fleurs_split(
            lang=lang,
            config=info["config"],
            split="train",
            target_needed=needed,
            existing_transcripts=existing_transcripts,
            current_saved=0
        )
        lang_entries.extend(train_entries)

        # If train split did not yield enough unique clips, pull from test split
        remaining = needed - len(lang_entries)
        if remaining > 0:
            logger.info(f"Pulling remaining {remaining} clips for {lang.upper()} from test split...")
            test_entries = stream_fleurs_split(
                lang=lang,
                config=info["config"],
                split="test",
                target_needed=remaining,
                existing_transcripts=existing_transcripts,
                current_saved=len(lang_entries)
            )
            lang_entries.extend(test_entries)

        logger.info(f"✅ Ingested {len(lang_entries)} new verified samples for {lang.upper()}!")
        all_new_entries.extend(lang_entries)

    # 3. Combine and save updated manifest_chunked.csv
    combined_rows = existing_rows + all_new_entries
    logger.info(f"\nSaving updated master manifest: {len(combined_rows)} total samples...")

    with open(MANIFEST_CHUNKED, "w", newline="", encoding="utf-8") as f:
        fieldnames = ["audio_path", "audio_name", "duration", "language", "transcript"]
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(combined_rows)

    # 4. Generate summary stats
    import pandas as pd
    df = pd.DataFrame(combined_rows)
    df["duration"] = pd.to_numeric(df["duration"], errors="coerce").fillna(0)
    print("\n" + "=" * 55)
    print("🎯 FINAL 20+ HOURS DATASET BREAKDOWN")
    print("=" * 55)
    summary = df.groupby("language")["duration"].agg(
        count="count",
        total_hours=lambda x: round(x.sum() / 3600, 2)
    )
    print(summary)
    total_samples = len(df)
    total_hours = round(df["duration"].sum() / 3600, 2)
    print(f"\nTOTAL SAMPLES : {total_samples}")
    print(f"TOTAL DURATION: {total_hours} Hours")
    print("=" * 55 + "\n")

    # 5. Package into colab_stt_dataset.zip
    create_training_zip(combined_rows)
    logger.info("🎉 20+ Hours Master Dataset build complete!")


if __name__ == "__main__":
    main()
