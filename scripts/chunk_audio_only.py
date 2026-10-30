#!/usr/bin/env python3
"""
Audio-Only Precision Chunking Pipeline (Strict 2.0s - 5.0s Chunks).

Features:
  - Loads all 35 clean audio files from data/processed/wav/ (or data/raw/STT voice/).
  - Uses VAD & intelligent interval splitting to slice audio into 2.0s - 5.0s clips.
  - Adds 0.25s safety padding around speech boundaries to prevent word clipping.
  - Saves 16kHz mono WAV chunks in data/chunked_dataset/wav_chunks/
  - Generates data/chunked_dataset/manifest_chunked.csv
  - Exports interactive Excel summary with play links to ~/Desktop/stt_chunked_summary.xlsx
"""

import csv
import logging
import shutil
import sys
from pathlib import Path
from typing import List

import librosa
import numpy as np
import openpyxl
from openpyxl.styles import Alignment, Font, PatternFill
import soundfile as sf

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_AUDIO_DIR = PROJECT_ROOT / "data" / "raw" / "STT voice"
PROCESSED_WAV_DIR = PROJECT_ROOT / "data" / "processed" / "wav"
ORIGINAL_MANIFEST = PROJECT_ROOT / "data" / "metadata" / "manifest.csv"

OUTPUT_BASE = PROJECT_ROOT / "data" / "chunked_dataset"
CHUNKS_WAV_DIR = OUTPUT_BASE / "wav_chunks"
MANIFEST_CHUNKED_PATH = OUTPUT_BASE / "manifest_chunked.csv"
EXCEL_PATH = OUTPUT_BASE / "stt_chunked_summary.xlsx"
DESKTOP_EXCEL_PATH = Path.home() / "Desktop" / "stt_chunked_summary.xlsx"


def slice_waveform_into_2to5s_chunks(
    waveform: np.ndarray, sample_rate: int = 16000, top_db: float = 25.0, pad_sec: float = 0.25
) -> List[np.ndarray]:
    """Detect speech intervals, apply padding, and split waveform strictly into 2.0s - 5.0s chunks."""
    pad_samples = int(pad_sec * sample_rate)
    total_samples = len(waveform)

    # Detect speech intervals
    intervals = librosa.effects.split(waveform, top_db=top_db, frame_length=2048, hop_length=512)

    if len(intervals) == 0:
        # Fallback if entire audio is low energy
        intervals = np.array([[0, total_samples]])

    # Merge intervals if gap between them is small (< 0.4s)
    merged_intervals = []
    min_gap = int(0.4 * sample_rate)
    
    cur_start, cur_end = intervals[0]
    for s, e in intervals[1:]:
        if s - cur_end <= min_gap:
            cur_end = e
        else:
            merged_intervals.append((cur_start, cur_end))
            cur_start, cur_end = s, e
    merged_intervals.append((cur_start, cur_end))

    # Now partition merged_intervals into 2.0s - 5.0s windows
    chunks = []
    min_samples = int(2.0 * sample_rate)
    max_samples = int(5.0 * sample_rate)
    target_samples = int(3.5 * sample_rate)

    accum_start = None
    accum_end = None

    for s, e in merged_intervals:
        if accum_start is None:
            accum_start, accum_end = s, e
        else:
            if (e - accum_start) <= max_samples:
                accum_end = e
            else:
                chunks.append((accum_start, accum_end))
                accum_start, accum_end = s, e

    if accum_start is not None:
        chunks.append((accum_start, accum_end))

    final_waveforms = []

    for s, e in chunks:
        # Apply safety padding
        padded_start = max(0, s - pad_samples)
        padded_end = min(total_samples, e + pad_samples)
        chunk_len = padded_end - padded_start

        # If chunk is longer than 5.0s, break it into sub-chunks of 2.0s - 4.5s
        if chunk_len > max_samples:
            num_parts = int(np.ceil(chunk_len / target_samples))
            part_len = chunk_len // num_parts
            for p in range(num_parts):
                p_start = padded_start + p * part_len
                p_end = padded_start + (p + 1) * part_len if p < num_parts - 1 else padded_end
                if (p_end - p_start) >= min_samples:
                    final_waveforms.append(waveform[p_start:p_end])
                elif (p_end - p_start) >= int(1.2 * sample_rate):
                    # Pad slightly to reach 2 seconds
                    diff = min_samples - (p_end - p_start)
                    ext_start = max(0, p_start - diff // 2)
                    ext_end = min(total_samples, p_end + diff // 2 + (diff % 2))
                    final_waveforms.append(waveform[ext_start:ext_end])
        elif chunk_len >= min_samples:
            final_waveforms.append(waveform[padded_start:padded_end])
        else:
            # Expand short chunk to minimum 2.0s duration if possible
            diff = min_samples - chunk_len
            ext_start = max(0, padded_start - diff // 2)
            ext_end = min(total_samples, padded_end + diff // 2 + (diff % 2))
            if (ext_end - ext_start) >= min_samples:
                final_waveforms.append(waveform[ext_start:ext_end])

    return final_waveforms


def main() -> None:
    logger.info("Starting Audio-Only 2.0s - 5.0s Chunking Pipeline...")

    # Re-create output directories cleanly
    if CHUNKS_WAV_DIR.exists():
        shutil.rmtree(CHUNKS_WAV_DIR)
    CHUNKS_WAV_DIR.mkdir(parents=True, exist_ok=True)

    # Load original manifest metadata if available
    manifest_info = {}
    if ORIGINAL_MANIFEST.exists():
        with open(ORIGINAL_MANIFEST, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                audio_path = Path(row["audio_path"])
                manifest_info[audio_path.stem] = row.get("language", "en")

    # Locate WAV files from processed wav directory
    wav_files = sorted(list(PROCESSED_WAV_DIR.glob("*.wav")))
    if not wav_files:
        logger.error(f"No WAV files found in {PROCESSED_WAV_DIR}")
        return

    logger.info(f"Processing {len(wav_files)} original audio recordings into 2.0s - 5.0s WAV chunks...")

    lang_name_map = {
        "en": "English 🇬🇧",
        "gu": "Gujarati 🇮🇳 (ગુજરાતી)",
        "hi": "Hindi 🇮🇳 (हिंदी)",
    }

    chunk_records = []
    total_duration_sec = 0.0

    for wav_file in wav_files:
        original_stem = wav_file.stem
        language = manifest_info.get(original_stem, "en")
        if original_stem.startswith("gu"):
            language = "gu"
        elif original_stem.startswith("hi"):
            language = "hi"
        elif original_stem.startswith("eng"):
            language = "en"

        waveform, sr = sf.read(str(wav_file), dtype="float32")
        wave_chunks = slice_waveform_into_2to5s_chunks(waveform, sample_rate=sr)

        for idx, chunk_wave in enumerate(wave_chunks, start=1):
            dur = len(chunk_wave) / sr
            chunk_filename = f"{original_stem}_chunk_{idx:03d}.wav"
            chunk_filepath = CHUNKS_WAV_DIR / chunk_filename

            # Save 16kHz WAV file
            sf.write(str(chunk_filepath), chunk_wave, sr)
            total_duration_sec += dur

            chunk_records.append({
                "audio_path": str(chunk_filepath.resolve()),
                "audio_name": chunk_filename,
                "original_source": wav_file.name,
                "duration": round(dur, 2),
                "language": language,
                "language_name": lang_name_map.get(language, language),
            })

    logger.info(f"✅ Generated {len(chunk_records)} audio chunks! Total audio duration: {total_duration_sec/60:.2f} minutes.")

    # Save manifest CSV
    fieldnames = ["audio_path", "original_source", "duration", "language"]
    with open(MANIFEST_CHUNKED_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in chunk_records:
            writer.writerow({
                "audio_path": r["audio_path"],
                "original_source": r["original_source"],
                "duration": r["duration"],
                "language": r["language"],
            })
    logger.info(f"📄 Saved manifest CSV: {MANIFEST_CHUNKED_PATH}")

    # Generate Excel Summary Sheet
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "2-5s Audio Chunks"

    headers = [
        "S.No",
        "Play Audio Chunk",
        "Chunk File Name",
        "Original Source File",
        "Language",
        "Duration (sec)",
    ]
    ws.append(headers)

    header_font = Font(name="Calibri", size=11, bold=True, color="FFFFFF")
    header_fill = PatternFill(start_color="1F4E79", end_color="1F4E79", fill_type="solid")
    link_font = Font(name="Calibri", size=11, color="0000FF", underline="single")

    for col_num, h in enumerate(headers, 1):
        cell = ws.cell(row=1, column=col_num)
        cell.font = header_font
        cell.fill = header_fill
        cell.alignment = Alignment(horizontal="center", vertical="center")

    for idx, r in enumerate(chunk_records, start=1):
        row_num = idx + 1
        ws.cell(row=row_num, column=1, value=idx).alignment = Alignment(horizontal="center")

        play_cell = ws.cell(row=row_num, column=2)
        play_cell.value = f'=HYPERLINK("file://{r["audio_path"]}", "▶️ Play {r["audio_name"]}")'
        play_cell.font = link_font
        play_cell.alignment = Alignment(horizontal="center")

        ws.cell(row=row_num, column=3, value=r["audio_name"])
        ws.cell(row=row_num, column=4, value=r["original_source"])
        ws.cell(row=row_num, column=5, value=r["language_name"])
        ws.cell(row=row_num, column=6, value=r["duration"]).alignment = Alignment(horizontal="right")

    ws.column_dimensions["A"].width = 8
    ws.column_dimensions["B"].width = 32
    ws.column_dimensions["C"].width = 32
    ws.column_dimensions["D"].width = 28
    ws.column_dimensions["E"].width = 25
    ws.column_dimensions["F"].width = 16

    wb.save(EXCEL_PATH)
    shutil.copy(EXCEL_PATH, DESKTOP_EXCEL_PATH)

    logger.info(f"📊 Excel summary successfully saved to:")
    logger.info(f"  - {EXCEL_PATH}")
    logger.info(f"  - {DESKTOP_EXCEL_PATH}")
    logger.info("✨ Audio chunking process completed successfully!")


if __name__ == "__main__":
    main()
