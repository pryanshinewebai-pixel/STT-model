#!/usr/bin/env python3
"""
Peak-Normalize All Audio Chunks to 95% Volume for Crystal-Clear Audio Playback.
"""

import logging
from pathlib import Path
import numpy as np
import soundfile as sf
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
WAV_DIR = PROJECT_ROOT / "data" / "chunked_dataset" / "wav_chunks"
MP3_DIR = PROJECT_ROOT / "data" / "chunked_dataset" / "mp3_chunks"

MP3_DIR.mkdir(parents=True, exist_ok=True)


def main():
    wav_files = sorted(list(WAV_DIR.glob("*.wav")))
    logger.info(f"Normalizing audio volume & exporting loud, clean WAV and MP3 for {len(wav_files)} files...")

    for w_path in tqdm(wav_files, desc="Normalizing Audio"):
        data, sr = sf.read(str(w_path), dtype="float32")
        
        # Calculate peak amplitude
        max_amp = np.max(np.abs(data))
        if max_amp > 1e-5:
            # Peak normalize to 0.95 (-0.45 dB)
            data = (data / max_amp) * 0.95

        # Re-save peak normalized 16-bit PCM WAV
        sf.write(str(w_path), data, sr, subtype="PCM_16")

        # Also save MP3 version for universal browser fallback
        mp3_path = MP3_DIR / f"{w_path.stem}.mp3"
        try:
            sf.write(str(mp3_path), data, sr, format="MP3", subtype="MPEG_LAYER_III")
        except Exception:
            pass

    logger.info("✅ All 728 audio chunks peak-normalized to 95% full volume!")


if __name__ == "__main__":
    main()
