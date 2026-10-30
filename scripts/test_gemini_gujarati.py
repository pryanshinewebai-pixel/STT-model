#!/usr/bin/env python3
import os
from pathlib import Path
from google import genai

client = genai.Client()

audio_path = Path("/Users/commerciax/proprietary_stt_model/data/processed/wav/guj1_5_clean.wav")

print(f"Uploading audio to Gemini: {audio_path.name}...")
uploaded_file = client.files.upload(file=audio_path)

print("Transcribing audio into native Gujarati script (ગુજરાતી)...")
response = client.models.generate_content(
    model="gemini-2.5-flash",
    contents=[
        uploaded_file,
        "Transcribe this audio file accurately. Write the transcript ONLY in native Gujarati script (ગુજરાતી લિપિ). Do not translate into English or Devanagari. Output only the transcript text."
    ]
)

print("\n" + "=" * 60)
print("🎧 AUDIO FILE:", audio_path.name)
print("📝 NATIVE GUJARATI SCRIPT (ગુજરાતી લિપિ):")
print("=" * 60)
print(response.text.strip())
print("=" * 60 + "\n")
