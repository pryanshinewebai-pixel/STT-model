#!/usr/bin/env python3
import speech_recognition as sr
from pathlib import Path

audio_path = Path("/Users/commerciax/proprietary_stt_model/data/processed/wav/guj1_5_clean.wav")

recognizer = sr.Recognizer()

print(f"Loading audio file: {audio_path.name}...")
with sr.AudioFile(str(audio_path)) as source:
    audio_data = recognizer.record(source)

print("Transcribing Gujarati audio into native Gujarati script (ગુજરાતી)...")
try:
    text = recognizer.recognize_google(audio_data, language="gu-IN")
    print("\n" + "=" * 60)
    print("🎧 AUDIO FILE:", audio_path.name)
    print("📝 NATIVE GUJARATI TRANSCRIPTION (ગુજરાતી લિપિ):")
    print("=" * 60)
    print(text)
    print("=" * 60 + "\n")
except Exception as e:
    print(f"Error transcribing: {e}")
