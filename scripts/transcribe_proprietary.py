#!/usr/bin/env python3
"""
CLI Tool — Transcribe audio files using the trained Proprietary STT Model with Language Model.

Usage:
    ./venv/bin/python scripts/transcribe_proprietary.py --audio "path/to/audio.m4a"
    ./venv/bin/python scripts/transcribe_proprietary.py --audio "path/to/audio.wav" --no-lm
"""

import argparse
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from inference.proprietary_inference import ProprietarySTTInference


def main():
    parser = argparse.ArgumentParser(description="Transcribe audio using Proprietary STT Model")
    parser.add_argument("--audio", type=str, required=True, help="Path to audio file (.wav, .m4a, .mp3)")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "mps", "cuda"], help="Inference device")
    parser.add_argument("--no-lm", action="store_true", help="Disable Language Model spelling correction")
    args = parser.parse_args()

    audio_path = Path(args.audio)
    if not audio_path.exists():
        print(f"❌ Error: Audio file not found at: {audio_path}")
        sys.exit(1)

    engine = ProprietarySTTInference(device=args.device)
    res = engine.transcribe(audio_path, use_lm=not args.no_lm)

    print("\n" + "=" * 60)
    print("🎙️  PROPRIETARY STT TRANSCRIPTION RESULT")
    print("=" * 60)
    print(f"File Name     : {res['file']}")
    print(f"Duration      : {res['duration']}s")
    print(f"Latency       : {res['latency_ms']} ms")
    print(f"Raw Output    : {res['raw_text']}")
    print(f"LM Final Text : {res['text']}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
