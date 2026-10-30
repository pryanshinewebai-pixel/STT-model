#!/usr/bin/env python3
"""
10+ Hours Multilingual Master STT Dataset Builder.

Expands dataset to 10+ Hours (~4,400 audio samples) across:
  - Gujarati (gu): ~3.8 hours (~1,370 samples)
  - Hindi (hi):    ~3.8 hours (~1,480 samples)
  - English (en):  ~3.8 hours (~1,540 samples)

Combines:
  1. 1,593 existing samples (all original chunks + 565 user mic recordings + early FLEURS)
  2. 2,800 new verified Google FLEURS samples (1,000 Gujarati + 900 Hindi + 900 English)

Outputs:
  - data/chunked_dataset/wav_chunks/ (16 kHz Mono WAVs, peak normalized to 0.95)
  - data/chunked_dataset/transcripts/ (.txt native script files)
  - data/chunked_dataset/manifest_chunked.csv (Master manifest)
  - /Users/commerciax/Desktop/colab_stt_dataset.zip (~700 MB ready for Google Colab GPU)
"""

import csv
import io
import json
import logging
import os
import shutil
import sys
import tarfile
import urllib.request
import zipfile
from pathlib import Path
from typing import Dict, List

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import requests
import soundfile as sf
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

DATA_BASE = PROJECT_ROOT / "data" / "chunked_dataset"
CHUNKS_WAV_DIR = DATA_BASE / "wav_chunks"
TRANSCRIPTS_DIR = DATA_BASE / "transcripts"
MANIFEST_CHUNKED = DATA_BASE / "manifest_chunked.csv"
DESKTOP_ZIP_PATH = Path.home() / "Desktop" / "colab_stt_dataset.zip"
LOCAL_ZIP_PATH = PROJECT_ROOT / "colab_stt_dataset.zip"

LANG_TARGETS = {
    "gu": {"config": "gu_in", "target_new": 1000, "name": "Gujarati 🇮🇳 (ગુજરાતી)"},
    "hi": {"config": "hi_in", "target_new": 900, "name": "Hindi 🇮🇳 (हिंदी)"},
    "en": {"config": "en_us", "target_new": 900, "name": "English 🇬🇧"},
}

HF_BASE_URL = "https://huggingface.co/datasets/google/fleurs/resolve/main/data"


def fetch_tsv_transcripts(config: str) -> Dict[str, str]:
    """Fetch train.tsv from Hugging Face and return a mapping of filename -> transcript."""
    url = f"{HF_BASE_URL}/{config}/train.tsv"
    logger.info(f"Downloading TSV metadata for {config} from {url}...")
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

    logger.info(f"Loaded {len(mapping)} verified transcripts for {config}.")
    return mapping


def stream_fleurs_samples(lang: str, config: str, target_count: int) -> List[Dict]:
    """Stream WAV files directly from train.tar.gz without downloading the entire multi-GB file."""
    tsv_map = fetch_tsv_transcripts(config)
    tar_url = f"{HF_BASE_URL}/{config}/audio/train.tar.gz"
    logger.info(f"Streaming {target_count} samples for {lang.upper()} ({config}) from {tar_url}...")

    headers = {"User-Agent": "Mozilla/5.0"}
    resp = requests.get(tar_url, stream=True, headers=headers, timeout=60)
    resp.raise_for_status()

    entries = []
    saved_count = 0

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

            saved_count += 1
            new_filename = f"fleurs_10h_{lang}_{saved_count:04d}.wav"
            target_wav_path = CHUNKS_WAV_DIR / new_filename
            target_txt_path = TRANSCRIPTS_DIR / f"fleurs_10h_{lang}_{saved_count:04d}.txt"

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

            if saved_count % 100 == 0 or saved_count == target_count:
                logger.info(f"Progress [{lang.upper()}]: {saved_count}/{target_count} samples saved.")

            if saved_count >= target_count:
                break

    resp.close()
    logger.info(f"✅ Successfully ingested {len(entries)} samples for {lang.upper()}!")
    return entries


def create_training_zip(manifest_rows: List[Dict]):
    """Packages all WAV files and manifest.csv into a unified colab_stt_dataset.zip."""
    logger.info("📦 Packaging 10+ Hour Dataset into colab_stt_dataset.zip...")

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
            # Add manifest.csv
            z.write(temp_manifest_path, arcname="manifest.csv")

            for r in tqdm(manifest_rows, desc="Zipping Audio Files"):
                wav_path = Path(r["audio_path"])
                if wav_path.exists():
                    z.write(wav_path, arcname=f"wav_chunks/{wav_path.name}")

        size_mb = os.path.getsize(zip_out) / (1024 * 1024)
        logger.info(f"🎉 Created {zip_out} ({size_mb:.2f} MB)")


def main():
    logger.info("🚀 Starting 10+ Hours Master STT Dataset Expansion...")

    # 1. Read existing manifest
    existing_rows = []
    if MANIFEST_CHUNKED.exists():
        with open(MANIFEST_CHUNKED, "r", encoding="utf-8") as f:
            existing_rows = list(csv.DictReader(f))
        logger.info(f"Loaded {len(existing_rows)} existing samples from {MANIFEST_CHUNKED}.")

    # Filter out any prior fleurs_10h_ runs if re-running
    existing_rows = [r for r in existing_rows if not Path(r["audio_path"]).name.startswith("fleurs_10h_")]
    logger.info(f"Base dataset has {len(existing_rows)} core samples.")

    # 2. Download and stream FLEURS for gu, hi, en
    all_new_entries = []
    for lang, info in LANG_TARGETS.items():
        new_entries = stream_fleurs_samples(lang, info["config"], info["target_new"])
        all_new_entries.extend(new_entries)

    # 3. Combine existing + new entries
    merged_rows = existing_rows + [
        {
            "audio_path": e["audio_path"],
            "duration": e["duration"],
            "language": e["language"],
            "transcript": e["transcript"],
        }
        for e in all_new_entries
    ]

    # 4. Save updated manifest_chunked.csv
    fieldnames = ["audio_path", "duration", "language", "transcript"]
    with open(MANIFEST_CHUNKED, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(merged_rows)

    logger.info(f"✅ Saved master manifest to {MANIFEST_CHUNKED} ({len(merged_rows)} total samples).")

    # 5. Calculate and display statistics
    stats = {}
    for r in merged_rows:
        lang = r["language"]
        dur = float(r["duration"])
        if lang not in stats:
            stats[lang] = {"count": 0, "total_sec": 0.0}
        stats[lang]["count"] += 1
        stats[lang]["total_sec"] += dur

    total_count = len(merged_rows)
    total_hours = sum(s["total_sec"] for s in stats.values()) / 3600.0

    print("\n" + "=" * 65)
    print("🏆 MASTER 10+ HOURS STT DATASET SUMMARY")
    print("=" * 65)
    print(f"{'Language':<25} {'Samples':<12} {'Duration (Hours)':<18}")
    print("-" * 65)
    for lang, s in stats.items():
        lang_label = LANG_TARGETS.get(lang, {}).get("name", lang)
        hrs = s["total_sec"] / 3600.0
        print(f"{lang_label:<25} {s['count']:<12} {hrs:.2f} hrs")
    print("-" * 65)
    print(f"{'TOTAL':<25} {total_count:<12} {total_hours:.2f} hrs")
    print("=" * 65 + "\n")

    # 6. Package colab_stt_dataset.zip
    create_training_zip(merged_rows)

    logger.info(f"🎉 10+ Hours Dataset successfully created!")
    logger.info(f"📁 Master Zip ready for Google Colab at: {DESKTOP_ZIP_PATH}")


if __name__ == "__main__":
    main()
