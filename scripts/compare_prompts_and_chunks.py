#!/usr/bin/env python3
"""
Evaluate Text Match & Similarity Percentage between Ground Truth STT Prompts & Chunked Transcripts.

Outputs:
  - Text Similarity Statistics (Word Match % & Character Match %)
  - Language-wise breakdown (English, Gujarati, Hindi)
  - Desktop Excel Report: ~/Desktop/stt_prompt_vs_chunk_comparison.xlsx
"""

import csv
import difflib
import logging
import re
import shutil
from pathlib import Path
from typing import Dict, List, Tuple

import openpyxl
from openpyxl.styles import Alignment, Font, PatternFill

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROMPTS_DIR = PROJECT_ROOT / "STT prompt"
MANIFEST_CHUNKED = PROJECT_ROOT / "data" / "chunked_dataset" / "manifest_chunked.csv"
OUTPUT_EXCEL = PROJECT_ROOT / "data" / "chunked_dataset" / "stt_prompt_vs_chunk_comparison.xlsx"
DESKTOP_EXCEL = Path.home() / "Desktop" / "stt_prompt_vs_chunk_comparison.xlsx"

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


def clean_str(s: str) -> str:
    """Normalize string for robust similarity matching."""
    s = s.lower()
    s = re.sub(r"[^\w\s\u0A80-\u0AFF\u0900-\u097F]", "", s)
    return re.sub(r"\s+", " ", s).strip()


def compute_char_similarity(a: str, b: str) -> float:
    """Compute character-level Levenshtein similarity percentage."""
    ca, cb = clean_str(a), clean_str(b)
    if not ca or not cb:
        return 0.0
    return round(difflib.SequenceMatcher(None, ca, cb).ratio() * 100.0, 2)


def compute_word_similarity(a: str, b: str) -> float:
    """Compute word-level overlap similarity percentage."""
    wa = clean_str(a).split()
    wb = clean_str(b).split()
    if not wa or not wb:
        return 0.0
    matcher = difflib.SequenceMatcher(None, wa, wb)
    return round(matcher.ratio() * 100.0, 2)


def main():
    logger.info("Starting Ground Truth STT Prompts vs Chunk Transcripts Comparison...")

    # Load ground truth prompts
    gt_prompts = {"en": [], "gu": [], "hi": []}
    for file_name, lang in PROMPT_FILE_LANG.items():
        file_path = PROMPTS_DIR / file_name
        if not file_path.exists():
            continue
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                line_str = line.strip()
                m = re.match(r"^\d+\.\s*(.+)$", line_str)
                if m:
                    gt_prompts[lang].append(m.group(1).strip())

    logger.info(f"Loaded ground truth prompts: EN={len(gt_prompts['en'])}, GU={len(gt_prompts['gu'])}, HI={len(gt_prompts['hi'])}")

    # Load chunk dataset manifest
    if not MANIFEST_CHUNKED.exists():
        logger.error(f"Manifest missing: {MANIFEST_CHUNKED}")
        return

    with open(MANIFEST_CHUNKED, "r", encoding="utf-8") as f:
        chunks = list(csv.DictReader(f))

    logger.info(f"Loaded {len(chunks)} chunk entries for comparison.")

    results = []
    stats_by_lang = {
        "en": {"char_sims": [], "word_sims": []},
        "gu": {"char_sims": [], "word_sims": []},
        "hi": {"char_sims": [], "word_sims": []},
    }

    for c in chunks:
        audio_name = Path(c["audio_path"]).name
        lang = c["language"]
        transcript = c["transcript"]
        candidates = gt_prompts.get(lang, [])

        best_prompt = ""
        best_char_sim = 0.0
        best_word_sim = 0.0

        if transcript and not transcript.startswith("[") and candidates:
            # Fast best prompt search
            for p in candidates:
                c_sim = compute_char_similarity(transcript, p)
                w_sim = compute_word_similarity(transcript, p)
                # Combine char & word similarity for best match criteria
                combo = c_sim * 0.6 + w_sim * 0.4
                if combo > (best_char_sim * 0.6 + best_word_sim * 0.4):
                    best_char_sim = c_sim
                    best_word_sim = w_sim
                    best_prompt = p

        stats_by_lang[lang]["char_sims"].append(best_char_sim)
        stats_by_lang[lang]["word_sims"].append(best_word_sim)

        results.append({
            "chunk_name": audio_name,
            "original_source": c["original_source"],
            "language": lang,
            "language_name": LANG_NAME_MAP.get(lang, lang),
            "duration": c["duration"],
            "chunk_transcript": transcript,
            "best_prompt": best_prompt or "[No Direct Ground Truth Match]",
            "char_match_pct": best_char_sim,
            "word_match_pct": best_word_sim,
        })

    # Summary calculations
    logger.info("=== SUMMARY MATCH PERCENTAGE RESULTS ===")
    overall_char_all = []
    overall_word_all = []

    summary_rows = []
    for lang in ["en", "gu", "hi"]:
        chars = stats_by_lang[lang]["char_sims"]
        words = stats_by_lang[lang]["word_sims"]
        avg_char = sum(chars) / len(chars) if chars else 0.0
        avg_word = sum(words) / len(words) if words else 0.0
        overall_char_all.extend(chars)
        overall_word_all.extend(words)

        # Count high accuracy chunks (> 75% match)
        high_acc = sum(1 for c, w in zip(chars, words) if c >= 70.0 or w >= 70.0)
        high_acc_pct = (high_acc / len(chars) * 100.0) if chars else 0.0

        logger.info(f"{LANG_NAME_MAP[lang]}: Avg Char Match = {avg_char:.2f}%, Avg Word Match = {avg_word:.2f}% (High Accuracy Chunks: {high_acc}/{len(chars)} = {high_acc_pct:.1f}%)")

        summary_rows.append({
            "language": LANG_NAME_MAP[lang],
            "total_chunks": len(chars),
            "avg_char_match": round(avg_char, 2),
            "avg_word_match": round(avg_word, 2),
            "high_acc_chunks": f"{high_acc} ({high_acc_pct:.1f}%)",
        })

    ov_char = sum(overall_char_all) / len(overall_char_all) if overall_char_all else 0.0
    ov_word = sum(overall_word_all) / len(overall_word_all) if overall_word_all else 0.0
    ov_high = sum(1 for c, w in zip(overall_char_all, overall_word_all) if c >= 70.0 or w >= 70.0)
    ov_high_pct = (ov_high / len(overall_char_all) * 100.0) if overall_char_all else 0.0

    logger.info(f"🎯 OVERALL DATASET: Avg Char Match = {ov_char:.2f}%, Avg Word Match = {ov_word:.2f}% (High Accuracy: {ov_high}/{len(overall_char_all)} = {ov_high_pct:.1f}%)")

    # Generate Excel Report with formatting
    wb = openpyxl.Workbook()
    
    # Sheet 1: Executive Summary
    ws1 = wb.active
    ws1.title = "Match Percentage Summary"

    ws1.append(["Language Dataset", "Total Chunks", "Avg Character Match (%)", "Avg Word Match (%)", "High Accuracy Chunks (>70%)"])
    header_font = Font(name="Calibri", size=11, bold=True, color="FFFFFF")
    header_fill = PatternFill(start_color="1F4E79", end_color="1F4E79", fill_type="solid")

    for col in range(1, 6):
        cell = ws1.cell(row=1, column=col)
        cell.font = header_font
        cell.fill = header_fill
        cell.alignment = Alignment(horizontal="center", vertical="center")

    for r in summary_rows:
        ws1.append([r["language"], r["total_chunks"], r["avg_char_match"], r["avg_word_match"], r["high_acc_chunks"]])

    # Append overall row
    ws1.append(["OVERALL TOTAL", len(overall_char_all), round(ov_char, 2), round(ov_word, 2), f"{ov_high} ({ov_high_pct:.1f}%)"])
    total_row = ws1.row_dimensions[len(summary_rows) + 2]
    for col in range(1, 6):
        cell = ws1.cell(row=len(summary_rows) + 2, column=col)
        cell.font = Font(name="Calibri", size=11, bold=True, color="1F4E79")

    ws1.column_dimensions["A"].width = 28
    ws1.column_dimensions["B"].width = 16
    ws1.column_dimensions["C"].width = 24
    ws1.column_dimensions["D"].width = 22
    ws1.column_dimensions["E"].width = 28

    # Sheet 2: Chunk-by-Chunk Detailed Comparison
    ws2 = wb.create_sheet(title="Chunk-by-Chunk Comparison")
    headers2 = [
        "S.No",
        "Chunk File Name",
        "Language",
        "Duration (sec)",
        "Chunk Transcript (Recognized)",
        "Ground Truth STT Prompt Line",
        "Char Match (%)",
        "Word Match (%)",
    ]
    ws2.append(headers2)

    for col in range(1, 9):
        cell = ws2.cell(row=1, column=col)
        cell.font = header_font
        cell.fill = header_fill
        cell.alignment = Alignment(horizontal="center", vertical="center")

    for idx, r in enumerate(results, start=1):
        row_idx = idx + 1
        ws2.cell(row=row_idx, column=1, value=idx).alignment = Alignment(horizontal="center")
        ws2.cell(row=row_idx, column=2, value=r["chunk_name"])
        ws2.cell(row=row_idx, column=3, value=r["language_name"])
        ws2.cell(row=row_idx, column=4, value=float(r["duration"])).alignment = Alignment(horizontal="right")
        ws2.cell(row=row_idx, column=5, value=r["chunk_transcript"])
        ws2.cell(row=row_idx, column=6, value=r["best_prompt"])
        ws2.cell(row=row_idx, column=7, value=r["char_match_pct"]).alignment = Alignment(horizontal="right")
        ws2.cell(row=row_idx, column=8, value=r["word_match_pct"]).alignment = Alignment(horizontal="right")

    ws2.column_dimensions["A"].width = 8
    ws2.column_dimensions["B"].width = 32
    ws2.column_dimensions["C"].width = 25
    ws2.column_dimensions["D"].width = 15
    ws2.column_dimensions["E"].width = 65
    ws2.column_dimensions["F"].width = 65
    ws2.column_dimensions["G"].width = 16
    ws2.column_dimensions["H"].width = 16

    wb.save(OUTPUT_EXCEL)
    shutil.copy(OUTPUT_EXCEL, DESKTOP_EXCEL)

    logger.info(f"📊 Comparison Excel Report saved to: {DESKTOP_EXCEL}")


if __name__ == "__main__":
    main()
