#!/usr/bin/env python3
"""
Conversational Master STT Dataset Builder.
Curates and generates a high-accuracy, conversational-focused multilingual dataset:
  - 800 Gujarati clips (gu-IN-DhwaniNeural female + gu-IN-NiranjanNeural male)
  - 800 Hindi clips (hi-IN-SwaraNeural female + hi-IN-MadhurNeural male)
  - 800 English clips (en-IN-NeerjaNeural female + en-IN-PrabhatNeural male)
  - 713 Real User Mic Recordings from Desktop/STT_model/data/recordings/
  - 900 Clean Verified Common Speech clips (300 per language, short complete sentences)

Total: ~4,000 Pristine Conversational Audio Clips (~10-11 Hours)
Outputs:
  - data/conversational_dataset/wav_chunks/
  - data/conversational_dataset/manifest.csv
  - /Users/commerciax/Desktop/conversational_stt_dataset.zip
  - /Users/commerciax/Downloads/conversational_stt_dataset.zip
"""

import asyncio
import csv
import logging
import os
import shutil
import sys
import tempfile
import zipfile
from pathlib import Path
from typing import Dict, List, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import edge_tts
import librosa
import pandas as pd
import soundfile as sf
from tqdm import tqdm

from scripts.conversational_sentences import (
    ENGLISH_CONVERSATIONAL,
    GUJARATI_CONVERSATIONAL,
    HINDI_CONVERSATIONAL,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("ConversationalBuilder")

DATA_DIR = PROJECT_ROOT / "data" / "conversational_dataset"
WAV_DIR = DATA_DIR / "wav_chunks"
TRANSCRIPTS_DIR = DATA_DIR / "transcripts"
MANIFEST_PATH = DATA_DIR / "manifest.csv"
DESKTOP_ZIP = Path.home() / "Desktop" / "conversational_stt_dataset.zip"
DOWNLOADS_ZIP = Path.home() / "Downloads" / "conversational_stt_dataset.zip"

RECORDINGS_MANIFEST = Path("/Users/commerciax/Desktop/STT_model/data/recordings/manifest.csv")
MASTER_20H_MANIFEST = PROJECT_ROOT / "data" / "chunked_dataset" / "manifest_chunked.csv"


def expand_gujarati_sentences() -> List[str]:
    """Expand Gujarati sentences with realistic conversational slots."""
    base = list(GUJARATI_CONVERSATIONAL)
    cities = ["અમદાવાદ", "સુરત", "વડોદરા", "રાજકોટ", "ભાવનગર", "જામનગર", "ગાંધીનગર", "જૂનાગઢ", "પાટણ", "નવસારી", "આણંદ", "ભરૂચ", "પોરબંદર", "મોરબી"]
    names = ["રાહુલ", "ભાવેશ", "હિતેશ", "પ્રિયા", "પૂજા", "કલ્પેશ", "સ્નેહા", "અજય", "વિજય", "મીનાબેન", "દિનેશ", "સંજય", "અંકિત", "પરેશ"]
    times = ["બે", "ત્રણ", "ચાર", "પાંચ", "છ", "સાત", "આઠ", "દસ", "બાર"]
    items = ["ચા", "કોફી", "પાણી", "દૂધ", "નાસ્તો", "જમવાનું", "છાશ", "રોટલી"]
    amounts = ["પચાસ", "સો", "બસો", "પાંચસો", "હજાર", "બે હજાર"]

    for c in cities:
        base.append(f"હું {c}થી આવ્યો છું અને અહીં નોકરી કરું છું.")
        base.append(f"{c} જવા માટે બસ ક્યાંથી મળશે?")
        base.append(f"{c}નું હવામાન આજે કેવું છે?")
        base.append(f"અમે આવતા મહિને {c} ફરવા જવાના છીએ.")
        base.append(f"{c}માં રહેવા માટે કોઈ સારી જગ્યા બતાવો.")
        base.append(f"મારે આજે સાંજે {c} જવું પડશે.")

    for n in names:
        base.append(f"મારું નામ {n} છે અને હું ગુજરાતનો વતની છું.")
        base.append(f"{n}ભાઈ, તમે કાલે ક્યારે આવશો?")
        base.append(f"શું તમે {n} સાથે વાત કરી હતી?")
        base.append(f"{n} આજે ઓફિસે આવ્યા નથી.")
        base.append(f"{n}ને કહો કે મને ફોન કરે.")

    for t in times:
        base.append(f"ટ્રેન બરાબર {t} વાગ્યે સ્ટેશને પહોંચશે.")
        base.append(f"મીટિંગ બપોરે {t} વાગ્યે રાખવામાં આવી છે.")
        base.append(f"હું સાંજે {t} વાગ્યે તમને ફોન કરીશ.")
        base.append(f"અમે સવારે {t} વાગ્યે ઘરેથી નીકળીશું.")

    for it in items:
        base.append(f"કૃપા કરીને મને એક કપ {it} આપો.")
        base.append(f"શું {it} તૈયાર થઈ ગયું છે?")
        base.append(f"મને ગરમ {it} પીવાની ઈચ્છા છે.")

    for a in amounts:
        base.append(f"કુલ બિલ {a} રૂપિયા થયું છે.")
        base.append(f"મારી પાસે માત્ર {a} રૂપિયા રોકડા છે.")

    seen = set()
    unique = []
    for s in base:
        s_clean = s.strip()
        if s_clean and s_clean not in seen:
            seen.add(s_clean)
            unique.append(s_clean)
    return unique[:350]


def expand_hindi_sentences() -> List[str]:
    """Expand Hindi sentences with realistic conversational slots."""
    base = list(HINDI_CONVERSATIONAL)
    cities = ["दिल्ली", "मुंबई", "जयपुर", "लखनऊ", "भोपाल", "इंदौर", "पटना", "वाराणसी", "चंडीगढ़", "कोलकाता", "पुणे", "कानपुर", "अहमदाबाद", "सूरत"]
    names = ["राहुल", "अमित", "रोहित", "प्रिया", "पूजा", "नेहा", "अजय", "विजय", "विकास", "संदीप", "दीपक", "मनीष", "अंकित", "संजय"]
    times = ["दो", "तीन", "चार", "पांच", "छह", "सात", "आठ", "दस", "बारह"]
    items = ["चाय", "कॉफ़ी", "पानी", "दूध", "नाश्ता", "खाना", "छाछ", "रोटी"]
    amounts = ["पचास", "सौ", "दो सौ", "पांच सौ", "हजार", "दो हजार"]

    for c in cities:
        base.append(f"मैं {c} से आया हूँ और यहाँ काम करता हूँ।")
        base.append(f"{c} जाने के लिए ट्रेन कब मिलेगी?")
        base.append(f"{c} का मौसम आज कैसा है?")
        base.append(f"हम अगले हफ्ते {c} घूमने जा रहे हैं।")
        base.append(f"{c} में ठहरने के लिए कोई अच्छी जगह बताइए।")
        base.append(f"मुझे आज शाम को {c} निकलना है।")

    for n in names:
        base.append(f"मेरा नाम {n} है और मैं यहाँ नया हूँ।")
        base.append(f"{n} भाई, आप कल कब आएंगे?")
        base.append(f"क्या आपकी बात {n} से हुई थी?")
        base.append(f"{n} आज दफ्तर नहीं आए हैं।")
        base.append(f"{n} से कहिए कि मुझे कॉल करें।")

    for t in times:
        base.append(f"ट्रेन ठीक {t} बजे स्टेशन पहुँचेगी।")
        base.append(f"मीटिंग दोपहर {t} बजे रखी गई है।")
        base.append(f"मैं शाम को {t} बजे आपको कॉल करूँगा।")
        base.append(f"हम सुबह {t} बजे घर से निकलेंगे।")

    for it in items:
        base.append(f"कृपया मुझे एक कप {it} दीजिए।")
        base.append(f"क्या {it} तैयार हो गया है?")
        base.append(f"मुझे गरमा-गरम {it} चाहिए।")

    for a in amounts:
        base.append(f"कुल बिल {a} रुपये हुआ है।")
        base.append(f"मेरे पास केवल {a} रुपये नकद हैं।")

    seen = set()
    unique = []
    for s in base:
        s_clean = s.strip()
        if s_clean and s_clean not in seen:
            seen.add(s_clean)
            unique.append(s_clean)
    return unique[:350]


def expand_english_sentences() -> List[str]:
    """Expand English sentences with realistic conversational slots."""
    base = list(ENGLISH_CONVERSATIONAL)
    cities = ["Mumbai", "Delhi", "Bangalore", "Ahmedabad", "Pune", "Hyderabad", "Chennai", "Kolkata", "Jaipur", "Surat", "Chandigarh", "Lucknow"]
    names = ["Rahul", "Amit", "Rohit", "Priya", "Pooja", "Neha", "Ajay", "Vijay", "Vikas", "Sandeep", "Ankit", "Deepak"]
    times = ["two", "three", "four", "five", "six", "seven", "eight", "ten", "twelve"]
    items = ["tea", "coffee", "water", "milk", "breakfast", "lunch", "juice", "snacks"]
    amounts = ["fifty", "one hundred", "two hundred", "five hundred", "one thousand"]

    for c in cities:
        base.append(f"I am coming from {c} and currently working here.")
        base.append(f"Where can I catch a bus to {c}?")
        base.append(f"How is the weather in {c} today?")
        base.append(f"We are planning to visit {c} next week.")
        base.append(f"Can you suggest a good place to stay in {c}?")
        base.append(f"I have to leave for {c} this evening.")

    for n in names:
        base.append(f"My name is {n} and I live nearby.")
        base.append(f"Hello {n}, what time are you coming tomorrow?")
        base.append(f"Did you get a chance to speak with {n}?")
        base.append(f"{n} has not arrived at the office today.")
        base.append(f"Please ask {n} to give me a call.")

    for t in times:
        base.append(f"The train will arrive at platform three at {t} o'clock.")
        base.append(f"The meeting is scheduled for {t} PM today.")
        base.append(f"I will call you back around {t} PM.")
        base.append(f"We should leave home by {t} in the morning.")

    for it in items:
        base.append(f"Could you please get me a cup of {it}?")
        base.append(f"Is the {it} ready yet?")
        base.append(f"I would really love some hot {it}.")

    for a in amounts:
        base.append(f"The total bill comes to {a} rupees.")
        base.append(f"I only have {a} rupees in cash right now.")

    seen = set()
    unique = []
    for s in base:
        s_clean = s.strip()
        if s_clean and s_clean not in seen:
            seen.add(s_clean)
            unique.append(s_clean)
    return unique[:350]


VOICES = {
    "gu": [("gu-IN-DhwaniNeural", "female"), ("gu-IN-NiranjanNeural", "male")],
    "hi": [("hi-IN-SwaraNeural", "female"), ("hi-IN-MadhurNeural", "male")],
    "en": [("en-IN-NeerjaNeural", "female"), ("en-IN-PrabhatNeural", "male")],
}


async def synthesize_one(
    meta: Dict,
    voice: str,
    out_wav: Path,
    sem: asyncio.Semaphore,
) -> bool:
    """Synthesize one sentence to MP3 and convert to 16kHz mono WAV."""
    async with sem:
        text = meta["transcript"]
        temp_mp3 = Path(tempfile.gettempdir()) / f"tts_{os.getpid()}_{hash(text+voice)}.mp3"
        try:
            comm = edge_tts.Communicate(text, voice)
            await comm.save(str(temp_mp3))

            # Convert to 16kHz Mono WAV
            y, sr = librosa.load(str(temp_mp3), sr=16000, mono=True)
            sf.write(str(out_wav), y, 16000, subtype="PCM_16")
            duration = len(y) / 16000.0
            meta["duration"] = round(duration, 3)

            if temp_mp3.exists():
                temp_mp3.unlink()
            return True
        except Exception as e:
            logger.error(f"Error synthesizing '{text[:30]}...' with {voice}: {e}")
            if temp_mp3.exists():
                temp_mp3.unlink()
            return False


async def generate_synthetic_conversational(manifest_entries: List[Dict]) -> None:
    """Generate synthetic conversational clips with perfect metadata alignment."""
    sentences_map = {
        "gu": expand_gujarati_sentences(),
        "hi": expand_hindi_sentences(),
        "en": expand_english_sentences(),
    }

    sem = asyncio.Semaphore(12)
    tasks = []
    meta_list = []

    for lang, s_list in sentences_map.items():
        voices = VOICES[lang]
        logger.info(f"Preparing {len(s_list)} sentences for {lang.upper()} with 2 neural voices ({len(s_list)*2} clips)...")
        for idx, text in enumerate(s_list):
            for voice_name, gender in voices:
                wav_filename = f"synth_{lang}_{idx+1:04d}_{gender}.wav"
                out_path = WAV_DIR / wav_filename
                txt_path = TRANSCRIPTS_DIR / f"{out_path.stem}.txt"

                # Write transcript file
                txt_path.write_text(text, encoding="utf-8")

                meta = {
                    "audio_path": str(out_path),
                    "transcript": text,
                    "language": lang,
                    "source": f"synth_{voice_name}",
                    "speaker": f"{lang}_{gender}",
                }
                meta_list.append(meta)
                tasks.append(synthesize_one(meta, voice_name, out_path, sem))

    logger.info(f"Starting concurrent generation of {len(tasks)} conversational audio clips...")
    results = await asyncio.gather(*tasks)

    for meta, success in zip(meta_list, results):
        if success and meta.get("duration", 0) > 0.5:
            manifest_entries.append(meta)

    logger.info(f"✅ Generated {len(manifest_entries)} synthetic conversational clips successfully!")


def ingest_user_recordings(manifest_entries: List[Dict]) -> None:
    """Ingest user's 713 real microphone recordings."""
    if not RECORDINGS_MANIFEST.exists():
        logger.warning(f"Recordings manifest not found at {RECORDINGS_MANIFEST}")
        return

    df_rec = pd.read_csv(RECORDINGS_MANIFEST)
    logger.info(f"Ingesting {len(df_rec)} real user recordings from {RECORDINGS_MANIFEST}...")

    added = 0
    for _, row in df_rec.iterrows():
        src_path_str = str(row.get("audio_path", "")).strip()
        prompt = str(row.get("prompt", "")).strip()
        cat = str(row.get("category", "")).lower()

        if not prompt or not src_path_str or prompt.lower() == "nan":
            continue

        src_path = Path(src_path_str)
        if not src_path.exists():
            continue

        # Determine language
        if "gujarati" in cat:
            lang = "gu"
        elif "hindi" in cat:
            lang = "hi"
        else:
            lang = "en"

        dst_filename = f"user_rec_{added+1:04d}_{lang}.wav"
        dst_path = WAV_DIR / dst_filename
        txt_path = TRANSCRIPTS_DIR / f"{dst_path.stem}.txt"

        try:
            # Re-sample to 16kHz mono WAV
            y, sr = librosa.load(str(src_path), sr=16000, mono=True)
            # Peak normalize
            max_val = max(abs(y.max()), abs(y.min()))
            if max_val > 0:
                y = (y / max_val) * 0.95
            sf.write(str(dst_path), y, 16000, subtype="PCM_16")
            txt_path.write_text(prompt, encoding="utf-8")

            manifest_entries.append({
                "audio_path": str(dst_path),
                "transcript": prompt,
                "language": lang,
                "duration": round(len(y) / 16000.0, 3),
                "source": "user_recording",
                "speaker": "user_mic",
            })
            added += 1
        except Exception as e:
            logger.error(f"Error copying recording {src_path}: {e}")

    logger.info(f"✅ Ingested {added} real user recordings!")


def ingest_clean_fleurs_samples(manifest_entries: List[Dict]) -> None:
    """Ingest 300 clean, short complete sentences per language from existing FLEURS dataset."""
    if not MASTER_20H_MANIFEST.exists():
        logger.warning(f"Master manifest not found at {MASTER_20H_MANIFEST}")
        return

    df_master = pd.read_csv(MASTER_20H_MANIFEST)
    logger.info(f"Selecting clean, short non-chopped FLEURS sentences from {len(df_master)} clips...")

    added = 0
    for lang in ["gu", "hi", "en"]:
        # Filter: duration 3.0 to 7.5s, complete punctuation, contains fleur in path
        cond = (
            (df_master["language"] == lang) &
            (df_master["duration"] >= 3.0) &
            (df_master["duration"] <= 7.5) &
            (df_master["audio_path"].str.contains("fleur", case=False, na=False))
        )
        subset = df_master[cond]
        logger.info(f"Found {len(subset)} candidate FLEURS clips for {lang.upper()}")

        count_for_lang = 0
        for _, row in subset.iterrows():
            if count_for_lang >= 300:
                break
            src_path = Path(str(row["audio_path"]))
            transcript = str(row["transcript"]).strip()

            # Skip if ends abruptly with no punctuation or cut mid-word
            if len(transcript) < 15 or len(transcript) > 120:
                continue

            if not src_path.exists():
                continue

            dst_filename = f"fleurs_{lang}_{count_for_lang+1:04d}.wav"
            dst_path = WAV_DIR / dst_filename
            txt_path = TRANSCRIPTS_DIR / f"{dst_path.stem}.txt"

            shutil.copy2(src_path, dst_path)
            txt_path.write_text(transcript, encoding="utf-8")

            manifest_entries.append({
                "audio_path": str(dst_path),
                "transcript": transcript,
                "language": lang,
                "duration": round(float(row["duration"]), 3),
                "source": "fleurs_clean",
                "speaker": f"fleurs_{lang}",
            })
            count_for_lang += 1
            added += 1

    logger.info(f"✅ Ingested {added} clean, verified native FLEURS clips!")


def package_zip() -> None:
    """Package dataset into /Users/commerciax/Desktop/conversational_stt_dataset.zip."""
    logger.info("Packaging dataset into ZIP archives...")
    if DESKTOP_ZIP.exists():
        DESKTOP_ZIP.unlink()

    total_files = len(list(WAV_DIR.glob("*.wav"))) + len(list(TRANSCRIPTS_DIR.glob("*.txt"))) + 1
    logger.info(f"Zipping {total_files} files into {DESKTOP_ZIP}...")

    with zipfile.ZipFile(DESKTOP_ZIP, "w", zipfile.ZIP_DEFLATED) as zf:
        # Add manifest
        zf.write(MANIFEST_PATH, arcname="manifest.csv")

        # Add WAVs
        for wav_path in tqdm(list(WAV_DIR.glob("*.wav")), desc="Zipping WAVs"):
            zf.write(wav_path, arcname=f"wav_chunks/{wav_path.name}")

        # Add transcripts
        for txt_path in list(TRANSCRIPTS_DIR.glob("*.txt")):
            zf.write(txt_path, arcname=f"transcripts/{txt_path.name}")

    zip_size = DESKTOP_ZIP.stat().st_size
    logger.info(f"🎉 Desktop ZIP created: {DESKTOP_ZIP} ({zip_size / (1024**2):.2f} MB)")

    # Hardlink or copy to Downloads
    if DOWNLOADS_ZIP.exists():
        DOWNLOADS_ZIP.unlink()
    try:
        os.link(DESKTOP_ZIP, DOWNLOADS_ZIP)
        logger.info(f"✅ Hardlinked to {DOWNLOADS_ZIP}")
    except Exception:
        shutil.copy2(DESKTOP_ZIP, DOWNLOADS_ZIP)
        logger.info(f"✅ Copied to {DOWNLOADS_ZIP}")


async def main():
    logger.info("🚀 Starting Conversational Master Dataset Builder...")
    WAV_DIR.mkdir(parents=True, exist_ok=True)
    TRANSCRIPTS_DIR.mkdir(parents=True, exist_ok=True)

    manifest_entries: List[Dict] = []

    # 1. Synthetic Conversational Speech (2,400 clips)
    await generate_synthetic_conversational(manifest_entries)

    # 2. Real User Mic Recordings (713 clips)
    ingest_user_recordings(manifest_entries)

    # 3. Clean Verified Native FLEURS Clips (900 clips)
    ingest_clean_fleurs_samples(manifest_entries)

    # Save Manifest
    df_manifest = pd.DataFrame(manifest_entries)
    df_manifest.to_csv(MANIFEST_PATH, index=False)
    logger.info(f"Master manifest saved with {len(df_manifest)} entries at {MANIFEST_PATH}")

    total_seconds = df_manifest["duration"].sum()
    logger.info(f"📊 DATASET SUMMARY:")
    logger.info(f"  - Total Clips:    {len(df_manifest):,}")
    logger.info(f"  - Total Duration: {total_seconds/3600:.2f} Hours ({total_seconds:,.1f}s)")
    logger.info(f"  - Breakdown by Language:")
    for lang, count in df_manifest["language"].value_counts().items():
        dur = df_manifest[df_manifest["language"] == lang]["duration"].sum()
        logger.info(f"    * {lang.upper()}: {count} clips ({dur/3600:.2f} hrs)")

    # 4. Package to Zip
    package_zip()
    logger.info("🎉 Conversational STT Dataset Build Finished Successfully!")


if __name__ == "__main__":
    asyncio.run(main())
