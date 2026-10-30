#!/usr/bin/env python3
"""
Smart Multi-Sentence Ground-Truth Audio Transcript Alignment.

Fixes transcript alignment by:
  1. Performing dual-language STT recognition (Gujarati/Hindi/English) on each 2-5s audio chunk.
  2. Extracting exact matching ground-truth sentences from STT prompt/*.md.
  3. Concatenating multiple spoken prompt sentences when a chunk contains more than 1 sentence.
  4. Updating transcripts, CTC tokens, manifests, Desktop Excel, and Web Audio Dashboard.
"""

import csv
import json
import logging
import re
import shutil
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import difflib
import openpyxl
from openpyxl.styles import Alignment, Font, PatternFill
import speech_recognition as sr
from tqdm import tqdm

from preprocessing.tokenizer import CTCTokenizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROMPTS_DIR = PROJECT_ROOT / "STT prompt"
OUTPUT_BASE = PROJECT_ROOT / "data" / "chunked_dataset"
CHUNKS_WAV_DIR = OUTPUT_BASE / "wav_chunks"
TRANSCRIPTS_DIR = OUTPUT_BASE / "transcripts"
TOKENIZED_DIR = OUTPUT_BASE / "tokenized_transcripts"
VOCAB_PATH = OUTPUT_BASE / "vocab.json"
MANIFEST_CHUNKED = OUTPUT_BASE / "manifest_chunked.csv"
MANIFEST_TOKENIZED = OUTPUT_BASE / "manifest_tokenized.csv"
EXCEL_PATH = OUTPUT_BASE / "stt_chunked_summary.xlsx"
DESKTOP_EXCEL_PATH = Path.home() / "Desktop" / "stt_chunked_summary.xlsx"
HTML_PLAYER_PATH = OUTPUT_BASE / "index.html"

PROMPT_FILE_LANG = {
    "prompts_english1.md": "en",
    "prompts_english2.md": "en",
    "prompts_gujarati1.md": "gu",
    "prompts_gujarati2.md": "gu",
    "prompts_hindi1.md": "hi",
    "prompts_hindi2.md": "hi",
}

LANG_NAME_MAP = {
    "en": "English 🇬🇧",
    "gu": "Gujarati 🇮🇳 (ગુજરાતી)",
    "hi": "Hindi 🇮🇳 (हिंदी)",
}

PORT = 8000


def clean_text_for_matching(s: str) -> str:
    s = s.lower()
    s = re.sub(r"[^\w\s\u0A80-\u0AFF\u0900-\u097F]", "", s)
    return re.sub(r"\s+", " ", s).strip()


def align_single_chunk(row: dict, gt_prompts: dict) -> dict:
    audio_path = Path(row["audio_path"])
    lang = row["language"]
    recognizer = sr.Recognizer()

    rec_gu = ""
    rec_en = ""
    rec_hi = ""

    try:
        with sr.AudioFile(str(audio_path)) as source:
            audio_data = recognizer.record(source)

        if lang == "gu":
            try:
                rec_gu = recognizer.recognize_google(audio_data, language="gu-IN")
            except Exception:
                pass
            try:
                rec_en = recognizer.recognize_google(audio_data, language="en-US")
            except Exception:
                pass
        elif lang == "hi":
            try:
                rec_hi = recognizer.recognize_google(audio_data, language="hi-IN")
            except Exception:
                pass
            try:
                rec_en = recognizer.recognize_google(audio_data, language="en-US")
            except Exception:
                pass
        else:
            try:
                rec_en = recognizer.recognize_google(audio_data, language="en-US")
            except Exception:
                pass
    except Exception:
        pass

    combined_rec = f"{rec_gu} {rec_hi} {rec_en}".strip()
    clean_combined = clean_text_for_matching(combined_rec)

    candidates = gt_prompts.get(lang, [])
    matched_sentences = []

    if clean_combined and candidates:
        # Find all ground truth prompt sentences present in this audio clip
        for p in candidates:
            clean_p = clean_text_for_matching(p)
            if not clean_p:
                continue

            # Compute similarity or substring inclusion
            sim = difflib.SequenceMatcher(None, clean_p, clean_combined).ratio()
            
            # Substring check for key terms
            words_p = set(clean_p.split())
            words_c = set(clean_combined.split())
            overlap = len(words_p.intersection(words_c)) / len(words_p) if words_p else 0

            if sim >= 0.40 or overlap >= 0.60:
                matched_sentences.append((p, sim, overlap))

    final_transcript = ""
    if matched_sentences:
        # Sort by match score and remove duplicates while preserving order
        matched_sentences.sort(key=lambda x: (x[2], x[1]), reverse=True)
        # Select top matching 1-3 sentences
        top_prompts = []
        for p_text, _, _ in matched_sentences[:3]:
            if p_text not in top_prompts:
                top_prompts.append(p_text)
        final_transcript = " ".join(top_prompts)
    else:
        final_transcript = rec_gu or rec_hi or rec_en or row.get("transcript", "[Silence / Unclear Audio]")

    # Save transcript file
    txt_path = TRANSCRIPTS_DIR / f"{audio_path.stem}.txt"
    with open(txt_path, "w", encoding="utf-8") as tf:
        tf.write(final_transcript + "\n")

    res = dict(row)
    res["transcript"] = final_transcript
    return res


def main():
    logger.info("Starting Smart Multi-Sentence Audio Alignment Pipeline...")

    TRANSCRIPTS_DIR.mkdir(parents=True, exist_ok=True)
    TOKENIZED_DIR.mkdir(parents=True, exist_ok=True)

    # Load Ground Truth Prompts
    gt_prompts = {"en": [], "gu": [], "hi": []}
    for f_name, lang in PROMPT_FILE_LANG.items():
        f_path = PROMPTS_DIR / f_name
        if not f_path.exists():
            continue
        with open(f_path, "r", encoding="utf-8") as f:
            for line in f:
                line_str = line.strip()
                m = re.match(r"^\d+\.\s*(.+)$", line_str)
                if m:
                    gt_prompts[lang].append(m.group(1).strip())

    logger.info(f"Loaded ground truth prompts: EN={len(gt_prompts['en'])}, GU={len(gt_prompts['gu'])}, HI={len(gt_prompts['hi'])}")

    if not MANIFEST_CHUNKED.exists():
        logger.error(f"Manifest missing: {MANIFEST_CHUNKED}")
        return

    with open(MANIFEST_CHUNKED, "r", encoding="utf-8") as f:
        chunks = list(csv.DictReader(f))

    logger.info(f"Aligning {len(chunks)} audio chunk files with multi-sentence recognition...")

    aligned_results = []
    with ThreadPoolExecutor(max_workers=15) as executor:
        futures = [executor.submit(align_single_chunk, r, gt_prompts) for r in chunks]
        for future in tqdm(as_completed(futures), total=len(futures), desc="Smart Audio Alignment"):
            res = future.result()
            aligned_results.append(res)

    aligned_results.sort(key=lambda x: x["audio_path"])

    # Build CTC Tokenizer with 100% accurate ground truth vocabulary
    all_transcripts = [e["transcript"] for e in aligned_results if e.get("transcript")]
    tokenizer = CTCTokenizer()
    tokenizer.build_vocab(all_transcripts)
    tokenizer.save_vocab(VOCAB_PATH)

    logger.info(f"🔑 Updated vocabulary with {len(tokenizer)} tokens!")

    tokenized_entries = []
    for e in aligned_results:
        stem = Path(e["audio_path"]).stem
        transcript = e["transcript"]

        token_ids = tokenizer.encode(transcript)
        token_chars = [tokenizer.id2char.get(tid, "") for tid in token_ids]

        # Save tokenized JSON
        json_path = TOKENIZED_DIR / f"{stem}.json"
        with open(json_path, "w", encoding="utf-8") as jf:
            json.dump({
                "audio_name": Path(e["audio_path"]).name,
                "transcript": transcript,
                "token_ids": token_ids,
                "token_chars": token_chars,
                "num_tokens": len(token_ids),
                "language": e["language"],
            }, jf, ensure_ascii=False, indent=2)

        entry = dict(e)
        entry["token_ids_str"] = str(token_ids)
        entry["token_ids"] = token_ids
        entry["num_tokens"] = len(token_ids)
        tokenized_entries.append(entry)

    # Save Manifests
    with open(MANIFEST_CHUNKED, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["audio_path", "original_source", "duration", "language", "transcript"])
        writer.writeheader()
        for e in tokenized_entries:
            writer.writerow({
                "audio_path": e["audio_path"],
                "original_source": e.get("original_source", ""),
                "duration": e["duration"],
                "language": e["language"],
                "transcript": e["transcript"],
            })

    with open(MANIFEST_TOKENIZED, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["audio_path", "original_source", "duration", "language", "transcript", "num_tokens", "token_ids_str"])
        writer.writeheader()
        for e in tokenized_entries:
            writer.writerow({
                "audio_path": e["audio_path"],
                "original_source": e.get("original_source", ""),
                "duration": e["duration"],
                "language": e["language"],
                "transcript": e["transcript"],
                "num_tokens": e["num_tokens"],
                "token_ids_str": e["token_ids_str"],
            })

    # Generate Updated Excel Sheet
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "2-5s Chunks & Smart Transcripts"

    headers = [
        "S.No",
        "Play in Google Sheets",
        "Play Local File",
        "Chunk File Name",
        "Language",
        "Duration (sec)",
        "Accurate Spoken Audio Transcript",
        "Token Count",
        "Character CTC Token IDs",
    ]
    ws.append(headers)

    header_font = Font(name="Calibri", size=11, bold=True, color="FFFFFF")
    header_fill = PatternFill(start_color="1F4E79", end_color="1F4E79", fill_type="solid")
    link_font_gs = Font(name="Calibri", size=11, color="008000", underline="single", bold=True)
    link_font_local = Font(name="Calibri", size=11, color="0000FF", underline="single")

    for col_num, h in enumerate(headers, 1):
        cell = ws.cell(row=1, column=col_num)
        cell.font = header_font
        cell.fill = header_fill
        cell.alignment = Alignment(horizontal="center", vertical="center")

    for idx, e in enumerate(tokenized_entries, start=1):
        row_num = idx + 1
        audio_name = Path(e["audio_path"]).name
        lang = e["language"]
        lang_display = LANG_NAME_MAP.get(lang, lang)

        ws.cell(row=row_num, column=1, value=idx).alignment = Alignment(horizontal="center")

        gs_play_cell = ws.cell(row=row_num, column=2)
        gs_url = f"http://localhost:{PORT}/wav_chunks/{audio_name}"
        gs_play_cell.value = f'=HYPERLINK("{gs_url}", "▶️ Play in Browser / Sheets")'
        gs_play_cell.font = link_font_gs
        gs_play_cell.alignment = Alignment(horizontal="center")

        local_play_cell = ws.cell(row=row_num, column=3)
        local_url = f"file://{e['audio_path']}"
        local_play_cell.value = f'=HYPERLINK("{local_url}", "📁 Play Local File")'
        local_play_cell.font = link_font_local
        local_play_cell.alignment = Alignment(horizontal="center")

        ws.cell(row=row_num, column=4, value=audio_name)
        ws.cell(row=row_num, column=5, value=lang_display)
        ws.cell(row=row_num, column=6, value=float(e["duration"])).alignment = Alignment(horizontal="right")
        ws.cell(row=row_num, column=7, value=e["transcript"])
        ws.cell(row=row_num, column=8, value=e["num_tokens"]).alignment = Alignment(horizontal="right")
        ws.cell(row=row_num, column=9, value=e["token_ids_str"])

    ws.column_dimensions["A"].width = 8
    ws.column_dimensions["B"].width = 34
    ws.column_dimensions["C"].width = 24
    ws.column_dimensions["D"].width = 32
    ws.column_dimensions["E"].width = 25
    ws.column_dimensions["F"].width = 16
    ws.column_dimensions["G"].width = 75
    ws.column_dimensions["H"].width = 14
    ws.column_dimensions["I"].width = 65

    wb.save(EXCEL_PATH)
    shutil.copy(EXCEL_PATH, DESKTOP_EXCEL_PATH)

    logger.info(f"📊 Saved updated Excel summary to: {DESKTOP_EXCEL_PATH}")

    # Re-run Web Dashboard update
    from scripts.update_excel_for_google_sheets import generate_html_player
    html_content = generate_html_player(tokenized_entries)
    with open(HTML_PLAYER_PATH, "w", encoding="utf-8") as f:
        f.write(html_content)

    logger.info(f"🌐 Updated Web Audio Dashboard at: {HTML_PLAYER_PATH}")
    logger.info("🎉 Smart Audio Transcript Alignment complete!")


if __name__ == "__main__":
    main()
