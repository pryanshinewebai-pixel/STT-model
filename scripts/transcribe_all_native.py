#!/usr/bin/env python3
"""
Native Multilingual Auto-Transcription Script.

Transcribes 35 audio files:
  - English (en) -> English text
  - Gujarati (gu) -> Native Gujarati script (ગુજરાતી)
  - Hindi (hi)    -> Native Hindi script (हिंदी)

Saves individual .txt files in data/processed/transcripts/
and updates data/metadata/manifest.csv.
"""

import csv
import logging
import sys
from pathlib import Path

# Add project root directory to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import speech_recognition as sr
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    manifest_path = Path("data/metadata/manifest.csv")
    transcripts_dir = Path("data/processed/transcripts")
    transcripts_dir.mkdir(parents=True, exist_ok=True)

    if not manifest_path.exists():
        logger.error(f"Manifest file not found: {manifest_path}")
        return

    with open(manifest_path, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    logger.info(f"Loaded {len(rows)} files from {manifest_path}")
    logger.info("Starting native transcription (English -> English, Gujarati -> ગુજરાતી, Hindi -> हिंदी)...")

    recognizer = sr.Recognizer()

    # Map language codes to recognition locales
    locale_map = {
        "en": "en-US",
        "gu": "gu-IN",
        "hi": "hi-IN",
    }

    updated_rows = []
    success_count = 0

    for row in tqdm(rows, desc="Transcribing All Files"):
        audio_path = Path(row["audio_path"])
        lang = row.get("language", "en")
        locale = locale_map.get(lang, "en-US")

        txt_file_path = transcripts_dir / f"{audio_path.stem}.txt"

        if not audio_path.exists():
            logger.warning(f"Audio file missing: {audio_path}")
            updated_rows.append(row)
            continue

        try:
            with sr.AudioFile(str(audio_path)) as source:
                audio_data = recognizer.record(source)

            transcript = recognizer.recognize_google(audio_data, language=locale)
            transcript = transcript.strip()

            # Save individual .txt file
            with open(txt_file_path, "w", encoding="utf-8") as txt_f:
                txt_f.write(transcript + "\n")

            row["transcript"] = transcript
            success_count += 1
        except Exception as e:
            logger.warning(f"Could not transcribe {audio_path.name} ({lang}): {e}")
            # Keep existing transcript if available
            pass

        updated_rows.append(row)

    # Write updated rows to manifest.csv
    fieldnames = ["audio_path", "feature_path", "duration", "num_frames", "language", "transcript"]
    with open(manifest_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(updated_rows)

    logger.info(f"🎉 Completed native transcription for {success_count}/{len(rows)} files!")
    logger.info(f"📁 Individual .txt files saved in: {transcripts_dir}")
    logger.info(f"📄 Updated manifest saved in: {manifest_path}")


if __name__ == "__main__":
    main()
