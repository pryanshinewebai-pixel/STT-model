#!/usr/bin/env python3
"""
Excel Dataset Summary Generator with Clickable Playable Audio Links.

Creates an Excel spreadsheet (data/metadata/stt_dataset_summary.xlsx)
with:
  - Clickable audio play link (▶️ Play Audio)
  - Audio file details
  - Transcribed native text side-by-side
"""

import csv
import logging
from pathlib import Path

import openpyxl
from openpyxl.styles import Alignment, Font, PatternFill

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    manifest_path = Path("data/metadata/manifest.csv")
    excel_path = Path("data/metadata/stt_dataset_summary.xlsx")
    desktop_excel_path = Path.home() / "Desktop" / "stt_dataset_summary.xlsx"

    if not manifest_path.exists():
        logger.error(f"Manifest file not found: {manifest_path}")
        return

    with open(manifest_path, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    logger.info(f"Loaded {len(rows)} entries from {manifest_path}")

    lang_name_map = {
        "en": "English 🇬🇧",
        "gu": "Gujarati 🇮🇳 (ગુજરાતી)",
        "hi": "Hindi 🇮🇳 (हिंदी)",
    }

    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "STT Audio & Transcripts"

    # Header styling
    headers = [
        "S.No",
        "Play Audio (Click to Listen)",
        "Audio File Name",
        "Language",
        "Duration (sec)",
        "Transcribed Native Text",
    ]
    ws.append(headers)

    header_font = Font(name="Calibri", size=11, bold=True, color="FFFFFF")
    header_fill = PatternFill(start_color="1F4E79", end_color="1F4E79", fill_type="solid")

    for col_num, header in enumerate(headers, 1):
        cell = ws.cell(row=1, column=col_num)
        cell.font = header_font
        cell.fill = header_fill
        cell.alignment = Alignment(horizontal="center", vertical="center")

    link_font = Font(name="Calibri", size=11, color="0000FF", underline="single")

    for idx, r in enumerate(rows, start=1):
        audio_full_path = Path(r["audio_path"]).resolve()
        lang_code = r.get("language", "en")
        
        row_num = idx + 1
        
        ws.cell(row=row_num, column=1, value=idx).alignment = Alignment(horizontal="center")
        
        # Clickable Hyperlink formula for direct audio playback in Excel/Mac
        play_cell = ws.cell(row=row_num, column=2)
        play_cell.value = f'=HYPERLINK("file://{audio_full_path}", "▶️ Play {audio_full_path.name}")'
        play_cell.font = link_font
        play_cell.alignment = Alignment(horizontal="center")

        ws.cell(row=row_num, column=3, value=audio_full_path.name)
        ws.cell(row=row_num, column=4, value=lang_name_map.get(lang_code, lang_code))
        ws.cell(row=row_num, column=5, value=float(r.get("duration", 0))).alignment = Alignment(horizontal="right")
        ws.cell(row=row_num, column=6, value=r.get("transcript", ""))

    # Auto-fit column widths
    ws.column_dimensions["A"].width = 8
    ws.column_dimensions["B"].width = 30
    ws.column_dimensions["C"].width = 28
    ws.column_dimensions["D"].width = 25
    ws.column_dimensions["E"].width = 15
    ws.column_dimensions["F"].width = 80

    wb.save(excel_path)
    wb.save(desktop_excel_path)

    logger.info(f"🎉 Created interactive Excel spreadsheet with audio links at:\n  - {excel_path}\n  - {desktop_excel_path}")


if __name__ == "__main__":
    main()
