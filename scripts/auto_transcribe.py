#!/usr/bin/env python3
"""
Auto-Transcription Script using OpenAI Whisper.

Transcribes all processed .wav audio files in data/metadata/manifest.csv
for English (en), Hindi (hi), and Gujarati (gu), then updates manifest.csv.

Usage:
    python scripts/auto_transcribe.py --manifest_path data/metadata/manifest.csv --model_size small
"""

import argparse
import csv
import logging
import sys
from pathlib import Path

# Add project root directory to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import whisper
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Auto-transcribe audio files in manifest.csv using Whisper AI.")
    parser.add_argument("--manifest_path", type=str, default="data/metadata/manifest.csv", help="Path to manifest CSV.")
    parser.add_argument("--model_size", type=str, default="small", help="Whisper model size (tiny, base, small, medium).")
    args = parser.parse_args()

    manifest_path = Path(args.manifest_path)
    if not manifest_path.exists():
        logger.error(f"Manifest file not found: {manifest_path}")
        return

    logger.info(f"Loading Whisper model '{args.model_size}'...")
    model = whisper.load_model(args.model_size)

    with open(manifest_path, "r", encoding="utf-8") as f:
        reader = list(csv.DictReader(f))

    if not reader:
        logger.warning("Manifest file is empty.")
        return

    logger.info(f"Transcribing {len(reader)} audio files...")
    updated_rows = []

    for row in tqdm(reader, desc="Transcribing"):
        audio_path = row["audio_path"]
        lang = row.get("language", "en")

        # Map language codes to Whisper language names if needed
        lang_map = {"en": "english", "hi": "hindi", "gu": "gujarati"}
        whisper_lang = lang_map.get(lang, lang)

        try:
            import soundfile as sf
            audio_np, _ = sf.read(audio_path, dtype="float32")
            # Transcribe with Whisper (passing float32 numpy array directly, avoiding ffmpeg CLI dependency)
            res = model.transcribe(audio_np, language=whisper_lang, fp16=False)
            transcript = res["text"].strip()
            row["transcript"] = transcript
        except Exception as e:
            logger.warning(f"Failed to transcribe {audio_path}: {e}")
            row["transcript"] = ""

        updated_rows.append(row)

    # Write updated rows back to manifest.csv
    fieldnames = ["audio_path", "feature_path", "duration", "num_frames", "language", "transcript"]
    with open(manifest_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(updated_rows)

    logger.info(f"🎉 Auto-transcription completed successfully! Updated {manifest_path}")


if __name__ == "__main__":
    main()
