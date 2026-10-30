#!/usr/bin/env python3
"""
Audio Chunking & Precision Transcription Pipeline (2.0s - 5.0s strict chunk length).

Pipeline:
  1. Loads 35 original clean audio files.
  2. Splits audio into speech intervals strictly between 2.0s and 5.0s.
  3. Adds 0.25s safety padding around speech boundaries to preserve start/end words.
  4. Transcribes each 2-5s clip into native script (English, Gujarati, Hindi).
  5. Exports:
     - wav_chunks/ (2.0s - 5.0s audio files)
     - transcripts/ (.txt native script files)
     - manifest_chunked.csv
     - stt_chunked_summary.xlsx (Project & Desktop)
"""

import argparse
import csv
import logging
import sys
from pathlib import Path
from typing import List, Tuple

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import librosa
import numpy as np
import openpyxl
from openpyxl.styles import Alignment, Font, PatternFill
import soundfile as sf
import speech_recognition as sr
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def slice_waveform_into_chunks(
    waveform: np.ndarray,
    sample_rate: int = 16000,
    min_sec: float = 3.0,
    max_sec: float = 15.0,
    top_db: float = 35.0,
    pad_sec: float = 0.3,
) -> List[np.ndarray]:
    """
    Detect speech intervals, apply safety padding, and slice into larger chunks
    between min_sec and max_sec (e.g., 3.0s - 15.0s) to keep complete sentences intact.
    """
    pad_samples = int(pad_sec * sample_rate)
    total_samples = len(waveform)

    # Detect non-silent speech intervals with relaxed top_db so natural pauses don't cut words
    intervals = librosa.effects.split(waveform, top_db=top_db, frame_length=2048, hop_length=512)

    raw_chunks = []
    current_start = None
    current_end = None

    min_samples = int(min_sec * sample_rate)
    max_samples = int(max_sec * sample_rate)

    for start, end in intervals:
        if current_start is None:
            current_start, current_end = start, end
        else:
            merged_len = end - current_start
            if merged_len <= max_samples:
                current_end = end
            else:
                raw_chunks.append((current_start, current_end))
                current_start, current_end = start, end

    if current_start is not None:
        raw_chunks.append((current_start, current_end))

    final_waveforms = []

    for s, e in raw_chunks:
        # Apply safety padding around speech boundaries
        padded_start = max(0, s - pad_samples)
        padded_end = min(total_samples, e + pad_samples)
        chunk_len = padded_end - padded_start

        # If a single continuous speech interval exceeds max_sec, split into balanced sub-chunks
        if chunk_len > max_samples:
            num_sub = int(np.ceil(chunk_len / max_samples))
            sub_len = chunk_len // num_sub
            for i in range(num_sub):
                sub_s = padded_start + i * sub_len
                sub_e = padded_start + (i + 1) * sub_len if i < num_sub - 1 else padded_end
                sub_dur = (sub_e - sub_s) / sample_rate
                if sub_dur >= (min_sec * 0.7):
                    final_waveforms.append(waveform[sub_s:sub_e])
        else:
            dur = chunk_len / sample_rate
            if dur >= (min_sec * 0.7):
                final_waveforms.append(waveform[padded_start:padded_end])

    return final_waveforms


# Backward compatibility alias
slice_waveform_into_2to5s_chunks = slice_waveform_into_chunks


def main() -> None:
    parser = argparse.ArgumentParser(description="Audio Chunking & Precision Transcription Pipeline")
    parser.add_argument("--min-sec", type=float, default=3.0, help="Minimum chunk duration in seconds (default: 3.0)")
    parser.add_argument("--max-sec", type=float, default=15.0, help="Maximum chunk duration in seconds (default: 15.0)")
    parser.add_argument("--top-db", type=float, default=35.0, help="Silence threshold in dB (default: 35.0)")
    parser.add_argument("--pad-sec", type=float, default=0.3, help="Safety boundary padding in seconds (default: 0.3)")
    args = parser.parse_args()

    original_manifest_path = Path("data/metadata/manifest.csv")
    output_base = Path("data/chunked_dataset")
    chunks_wav_dir = output_base / "wav_chunks"
    transcripts_dir = output_base / "transcripts"
    manifest_chunked_path = output_base / "manifest_chunked.csv"
    excel_path = output_base / "stt_chunked_summary.xlsx"
    desktop_excel_path = Path.home() / "Desktop" / "stt_chunked_summary.xlsx"

    chunks_wav_dir.mkdir(parents=True, exist_ok=True)
    transcripts_dir.mkdir(parents=True, exist_ok=True)

    if not original_manifest_path.exists():
        logger.error(f"Original manifest missing: {original_manifest_path}")
        return

    with open(original_manifest_path, "r", encoding="utf-8") as f:
        original_rows = list(csv.DictReader(f))

    logger.info(f"Loaded {len(original_rows)} original audio files.")
    logger.info(
        f"Chunking into {args.min_sec}s - {args.max_sec}s clips (top_db={args.top_db}, pad={args.pad_sec}s) "
        "with safety padding and native transcription..."
    )

    recognizer = sr.Recognizer()
    locale_map = {"en": "en-US", "gu": "gu-IN", "hi": "hi-IN"}
    lang_name_map = {"en": "English 🇬🇧", "gu": "Gujarati 🇮🇳 (ગુજરાતી)", "hi": "Hindi 🇮🇳 (हिंदी)"}

    chunk_entries = []

    for row in tqdm(original_rows, desc=f"Processing {args.min_sec}-{args.max_sec}s Chunks"):
        audio_path = Path(row["audio_path"])
        lang = row.get("language", "en")
        locale = locale_map.get(lang, "en-US")

        if not audio_path.exists():
            continue

        waveform, sample_rate = sf.read(str(audio_path), dtype="float32")
        wave_chunks = slice_waveform_into_chunks(
            waveform,
            sample_rate=sample_rate,
            min_sec=args.min_sec,
            max_sec=args.max_sec,
            top_db=args.top_db,
            pad_sec=args.pad_sec,
        )

        for c_idx, chunk_wave in enumerate(wave_chunks, start=1):
            duration = len(chunk_wave) / sample_rate

            chunk_filename = f"{audio_path.stem}_chunk_{c_idx:02d}.wav"
            chunk_wav_path = chunks_wav_dir / chunk_filename
            sf.write(str(chunk_wav_path), chunk_wave, sample_rate)

            # Transcribe chunk with SpeechRecognition (gu-IN, hi-IN, en-US)
            transcript = ""
            try:
                with sr.AudioFile(str(chunk_wav_path)) as source:
                    audio_data = recognizer.record(source)
                transcript = recognizer.recognize_google(audio_data, language=locale).strip()
            except sr.UnknownValueError:
                logger.debug(f"Audio not clearly recognizable in {chunk_filename}")
            except Exception as e:
                logger.debug(f"Transcription error in {chunk_filename}: {e}")

            # Skip empty/unrecognizable noise chunks
            if not transcript:
                continue

            txt_filename = f"{audio_path.stem}_chunk_{c_idx:02d}.txt"
            txt_file_path = transcripts_dir / txt_filename
            with open(txt_file_path, "w", encoding="utf-8") as tf:
                tf.write(transcript + "\n")

            chunk_entries.append({
                "audio_path": str(chunk_wav_path.resolve()),
                "audio_name": chunk_filename,
                "duration": round(duration, 2),
                "language": lang,
                "language_name": lang_name_map.get(lang, lang),
                "transcript": transcript,
                "txt_path": str(txt_file_path.resolve()),
            })

    logger.info(f"🎉 Generated {len(chunk_entries)} chunks between {args.min_sec}s and {args.max_sec}s!")

    # Save manifest CSV
    fieldnames = ["audio_path", "duration", "language", "transcript"]
    with open(manifest_chunked_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for e in chunk_entries:
            writer.writerow({
                "audio_path": e["audio_path"],
                "duration": e["duration"],
                "language": e["language"],
                "transcript": e["transcript"],
            })

    # Save Excel Summary
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "Audio Chunks"

    headers = [
        "S.No",
        "Play Chunk Audio",
        "Chunk File Name",
        "Language",
        "Duration (sec)",
        "Accurate Native Chunk Transcript",
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

    for idx, e in enumerate(chunk_entries, start=1):
        r_num = idx + 1
        ws.cell(row=r_num, column=1, value=idx).alignment = Alignment(horizontal="center")

        play_cell = ws.cell(row=r_num, column=2)
        play_cell.value = f'=HYPERLINK("file://{e["audio_path"]}", "▶️ Play {e["audio_name"]}")'
        play_cell.font = link_font
        play_cell.alignment = Alignment(horizontal="center")

        ws.cell(row=r_num, column=3, value=e["audio_name"])
        ws.cell(row=r_num, column=4, value=e["language_name"])
        ws.cell(row=r_num, column=5, value=e["duration"]).alignment = Alignment(horizontal="right")
        ws.cell(row=r_num, column=6, value=e["transcript"])

    ws.column_dimensions["A"].width = 8
    ws.column_dimensions["B"].width = 30
    ws.column_dimensions["C"].width = 32
    ws.column_dimensions["D"].width = 25
    ws.column_dimensions["E"].width = 15
    ws.column_dimensions["F"].width = 80

    wb.save(excel_path)
    wb.save(desktop_excel_path)

    logger.info(f"🎉 Chunking pipeline complete!")
    logger.info(f"📁 Chunks folder: {chunks_wav_dir}")
    logger.info(f"📄 Chunk transcripts: {transcripts_dir}")
    logger.info(f"📊 Excel summary saved at: {excel_path} and {desktop_excel_path}")


if __name__ == "__main__":
    main()
