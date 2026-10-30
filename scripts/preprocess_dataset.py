#!/usr/bin/env python3
"""
Batch Preprocessing Script — Processes raw audio files into clean WAVs, feature tensors & manifest.

Pipeline:
  1. Load raw audio (M4A/WAV/MP3/FLAC)
  2. Resample to 16 kHz
  3. Convert to Mono
  4. Trim leading and trailing silence
  5. Peak-normalize amplitude [-1.0, 1.0]
  6. Export clean .wav file
  7. Extract Log-Mel Spectrogram feature tensor (.pt)
  8. Build manifest.csv with language tags (en, hi, gu)

Usage:
    python scripts/preprocess_dataset.py --input_dir "data/raw/STT voice" --output_dir data/processed --manifest_path data/metadata/manifest.csv
"""

import argparse
import csv
import logging
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional

# Add project root directory to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import soundfile as sf
import torch
import yaml
from tqdm import tqdm

from preprocessing.audio import AudioProcessor
from preprocessing.features import FeatureExtractor

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> Dict[str, Any]:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def detect_language(path: Path) -> str:
    """Infer language code (en, hi, gu) from directory path or filename."""
    path_str = str(path).lower()
    if "english" in path_str or "eng" in path_str:
        return "en"
    elif "hindi" in path_str or "hin" in path_str:
        return "hi"
    elif "gujrati" in path_str or "gujarati" in path_str or "guj" in path_str:
        return "gu"
    return "en"


def process_single_file(
    audio_path: Path,
    output_dir: Path,
    audio_cfg: Dict[str, Any],
    feat_cfg: Dict[str, Any],
    override_language: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """Process a single audio file: resample, mono, trim, normalize, save WAV + feature tensor."""
    try:
        wav_out_dir = output_dir / "wav"
        feat_out_dir = output_dir / "features"

        wav_out_dir.mkdir(parents=True, exist_ok=True)
        feat_out_dir.mkdir(parents=True, exist_ok=True)

        processor = AudioProcessor(
            sample_rate=audio_cfg.get("sample_rate", 16000),
            mono=audio_cfg.get("mono", True),
            max_duration=audio_cfg.get("max_duration_seconds", 300.0),
            min_duration=audio_cfg.get("min_duration_seconds", 0.1),
            trim_silence=True,
        )
        extractor = FeatureExtractor(
            n_mels=feat_cfg.get("n_mels", 80),
            n_fft=feat_cfg.get("n_fft", 400),
            hop_length=feat_cfg.get("hop_length", 160),
            win_length=feat_cfg.get("win_length", 400),
            f_min=feat_cfg.get("f_min", 0.0),
            f_max=feat_cfg.get("f_max", 8000.0),
            normalize=feat_cfg.get("normalize", True),
        )

        waveform, sr = processor.load(audio_path)
        duration = len(waveform) / sr

        # Save clean processed WAV file (16kHz, mono, trimmed, normalized)
        cleaned_wav_path = wav_out_dir / f"{audio_path.stem}_clean.wav"
        sf.write(str(cleaned_wav_path), waveform, sr)

        # Compute Log-Mel Spectrogram feature tensor
        features = extractor.compute_mel_spectrogram(waveform, sample_rate=sr)

        # Save PyTorch feature tensor (.pt)
        feature_path = feat_out_dir / f"{audio_path.stem}_feat.pt"
        torch.save(features, feature_path)

        lang = override_language if override_language else detect_language(audio_path)

        return {
            "audio_path": str(cleaned_wav_path.resolve()),
            "feature_path": str(feature_path.resolve()),
            "duration": round(duration, 3),
            "num_frames": features.shape[-1],
            "language": lang,
            "transcript": "",  # Placeholder for transcription text
        }
    except Exception as e:
        logger.warning(f"Skipping {audio_path.name}: {e}")
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch process audio files into clean WAVs and log-mel feature tensors.")
    parser.add_argument("--input_dir", type=str, default="data/raw/STT voice", help="Directory containing raw audio files.")
    parser.add_argument("--output_dir", type=str, default="data/processed", help="Output directory for clean WAVs and .pt feature tensors.")
    parser.add_argument("--manifest_path", type=str, default="data/metadata/manifest.csv", help="Path to save output manifest CSV.")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Path to YAML configuration file.")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of parallel worker processes.")
    parser.add_argument("--language", type=str, default=None, help="Override language code for all files (en, hi, gu).")

    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    manifest_path = Path(args.manifest_path)

    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    if not input_dir.exists():
        logger.error(f"Input directory does not exist: {input_dir}")
        return

    config = load_config(args.config)
    audio_cfg = config.get("audio", {})
    feat_cfg = config.get("features", {})

    # Support multiple audio extensions
    audio_extensions = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}
    audio_files = sorted([p for p in input_dir.rglob("*") if p.suffix.lower() in audio_extensions])

    if not audio_files:
        logger.warning(f"No supported audio files found in {input_dir}")
        return

    logger.info(f"Found {len(audio_files)} audio files in '{input_dir}'. Starting batch processing...")

    manifest_entries: List[Dict[str, Any]] = []
    skipped_count = 0

    with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
        futures = {
            executor.submit(
                process_single_file,
                f,
                output_dir,
                audio_cfg,
                feat_cfg,
                args.language,
            ): f
            for f in audio_files
        }

        for future in tqdm(as_completed(futures), total=len(futures), desc="Preprocessing audio"):
            result = future.result()
            if result is not None:
                manifest_entries.append(result)
            else:
                skipped_count += 1

    # Sort manifest entries by audio path for deterministic order
    manifest_entries.sort(key=lambda x: x["audio_path"])

    # Write manifest CSV
    fieldnames = ["audio_path", "feature_path", "duration", "num_frames", "language", "transcript"]
    with open(manifest_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(manifest_entries)

    logger.info(f"🎉 Processing complete!")
    logger.info(f"Successfully processed: {len(manifest_entries)} files")
    logger.info(f"Skipped: {skipped_count} files")
    logger.info(f"Cleaned WAVs saved in: {output_dir / 'wav'}")
    logger.info(f"Feature tensors saved in: {output_dir / 'features'}")
    logger.info(f"Manifest written to: {manifest_path}")


if __name__ == "__main__":
    main()
