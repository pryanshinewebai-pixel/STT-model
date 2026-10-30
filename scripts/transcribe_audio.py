#!/usr/bin/env python3
"""
CLI Tool — Transcribe Audio Files with Fine-Tuned Multilingual STT Model.

Usage:
    ./venv/bin/python scripts/transcribe_audio.py --audio path/to/sample.wav
    ./venv/bin/python scripts/transcribe_audio.py --audio path/to/sample.wav --lang gu
"""

import argparse
import logging
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from inference.inference import STTInference

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Transcribe audio files using Fine-Tuned Multilingual STT Model.")
    parser.add_argument("--audio", type=str, required=True, help="Path to audio file (.wav, .mp3, .flac).")
    parser.add_argument("--lang", type=str, default="auto", choices=["auto", "en", "hi", "gu"], help="Language hint (default: auto).")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "mps", "cuda", "auto"], help="Device to run inference on.")
    args = parser.parse_args()

    audio_path = Path(args.audio)
    if not audio_path.exists():
        print(f"❌ Error: Audio file not found at: {audio_path}")
        sys.exit(1)

    engine = STTInference(device=args.device)

    print(f"\n🎧 Transcribing: {audio_path.name}")
    print(f"🌐 Target Language: {args.lang.upper() if args.lang != 'auto' else 'Auto-Detect'}")

    lang_param = None if args.lang == "auto" else args.lang
    result = engine.transcribe_file(audio_path, language=lang_param)

    print("\n" + "=" * 60)
    print("🎙️ TRANSCRIPTION RESULT")
    print("=" * 60)
    print(f"File Name     : {result['file_name']}")
    print(f"Audio Duration: {result['duration']} seconds")
    print(f"Detected Lang : {result['language']}")
    print(f"Transcription : {result['text']}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
