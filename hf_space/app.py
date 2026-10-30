import os
import re
import time
from pathlib import Path
import gradio as gr
import librosa
import numpy as np
import soundfile as sf
import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor

# --- Model Loading ---
MODEL_DIR = Path(__file__).resolve().parent / "model"
if not MODEL_DIR.exists() or not (MODEL_DIR / "model.safetensors").exists():
    MODEL_NAME = "openai/whisper-base"
else:
    MODEL_NAME = str(MODEL_DIR)

print(f"Loading STT Model from: {MODEL_NAME}...")
device = "cuda" if torch.cuda.is_available() else "cpu"
processor = WhisperProcessor.from_pretrained(MODEL_NAME)
model = WhisperForConditionalGeneration.from_pretrained(MODEL_NAME).to(device)
model.eval()
print(f"✅ Model loaded successfully on {device}!")

# Language Mappings
LANG_CODES = {
    "Gujarati (ગુજરાતી)": "gu",
    "Hindi (हिंदी)": "hi",
    "English": "en",
    "Auto Detect": None,
}


def clean_gujarati_orthography(text: str) -> str:
    """Cleans common ASR phonetic variations and conjuncts for conversational Gujarati."""
    if not text:
        return text

    replacements = [
        ("નવાસ્કાર", "નમસ્કાર"),
        ("નમસકાર", "નમસ્કાર"),
        ("દમસકાર", "નમસ્કાર"),
        ("દમસ્કાર", "નમસ્કાર"),
        ("વારો નામ", "મારું નામ"),
        ("મારુ નામ", "મારું નામ"),
        ("મારુનામ", "મારું નામ"),
        ("તમારુનામ", "તમારું નામ"),
        ("તમારુ નામ", "તમારું નામ"),
        ("મજોના", "મજામાં"),
        ("મજામાત", "મજામાં"),
        ("મજામાંં", "મજામાં"),
        ("મજામા", "મજામાં"),
        ("ક્યાથી", "ક્યાંથી"),
        ("ક્યાંથીં", "ક્યાંથી"),
        ("આવ્યાચો", "આવ્યા છો"),
        ("આવિયા છો", "આવ્યા છો"),
        ("શૂટ છે", "શું છે"),
        ("શુ છે", "શું છે"),
        ("શુ ", "શું "),
        ("હુ ", "હું "),
    ]
    for old, new in replacements:
        text = text.replace(old, new)

    # Strip hallucinated trailing fragments
    text = re.sub(r"[\s,\.]+(?:તમારો|અમારો|અમે|તમે)\s+[અ-હ]{1,3}\.*$", "", text)
    text = re.sub(r"[ં]+", "ં", text)
    return text.strip()


def trim_silence(audio: np.ndarray, sr: int = 16000, top_db: float = 28.0) -> np.ndarray:
    """Trims leading and trailing silence to avoid hallucinations."""
    if len(audio) == 0:
        return audio
    frame_len = int(sr * 0.030)
    hop_len = int(sr * 0.010)
    if len(audio) < frame_len:
        return audio

    energy = np.array([
        np.sqrt(np.mean(audio[i : i + frame_len] ** 2))
        for i in range(0, len(audio) - frame_len, hop_len)
    ])
    max_e = np.max(energy)
    if max_e < 1e-4:
        return audio

    threshold = max_e * (10 ** (-top_db / 20))
    speech_indices = np.where(energy > threshold)[0]
    if len(speech_indices) == 0:
        return audio

    start_frame = speech_indices[0]
    end_frame = speech_indices[-1]

    # Ignore trailing clicks
    for k in range(len(speech_indices) - 1, 0, -1):
        gap = speech_indices[k] - speech_indices[k - 1]
        if gap > 50:
            burst_len = speech_indices[-1] - speech_indices[k] + 1
            if burst_len < 12:
                end_frame = speech_indices[k - 1]
            break

    start_idx = max(0, start_frame * hop_len - int(sr * 0.12))
    end_idx = min(len(audio), (end_frame + 1) * hop_len + frame_len + int(sr * 0.12))
    return audio[start_idx:end_idx]


def transcribe(audio_path: str, language_choice: str):
    if not audio_path:
        return "⚠️ કૃપા કરીને માઇક્રોફોનથી બોલો અથવા ઓડિયો ફાઇલ અપલોડ કરો.", "No audio detected"

    t0 = time.time()
    try:
        # Load audio and convert to 16kHz mono
        audio, sr = librosa.load(audio_path, sr=16000, mono=True)
        audio = trim_silence(audio, sr=16000)

        # Peak normalization
        max_amp = np.max(np.abs(audio)) if len(audio) > 0 else 0
        if max_amp > 0.005:
            audio = (audio / max_amp) * 0.95

        input_features = processor(audio, sampling_rate=16000, return_tensors="pt").input_features.to(device)

        lang_code = LANG_CODES.get(language_choice)
        gen_kwargs = {
            "max_new_tokens": 128,
            "num_beams": 4,
            "repetition_penalty": 1.15,
            "task": "transcribe",
        }

        if lang_code:
            forced_ids = processor.get_decoder_prompt_ids(language=lang_code, task="transcribe")
            gen_kwargs["forced_decoder_ids"] = forced_ids

        with torch.no_grad():
            pred_ids = model.generate(input_features, **gen_kwargs)

        transcription = processor.batch_decode(pred_ids, skip_special_tokens=True)[0].strip()

        # Clean Gujarati orthography
        if lang_code == "gu" or (lang_code is None and any("\u0A80" <= c <= "\u0AFF" for c in transcription)):
            transcription = clean_gujarati_orthography(transcription)

        elapsed = round(time.time() - t0, 2)
        status_msg = f"⏱️ Transcribed in {elapsed}s | Language: {language_choice} | Device: {device.upper()}"
        return transcription, status_msg

    except Exception as e:
        return f"❌ Error: {str(e)}", "Inference Failed"


# --- Gradio UI Layout ---
custom_css = """
.container { max-width: 850px; margin: auto; padding: 20px; }
.header-title { text-align: center; font-size: 28px; font-weight: bold; margin-bottom: 5px; }
.header-desc { text-align: center; color: #666; font-size: 15px; margin-bottom: 25px; }
"""

with gr.Blocks(title="Indic Speech-to-Text AI", css=custom_css, theme=gr.themes.Soft()) as demo:
    gr.HTML(
        """
        <div class="header-title">🎙️ Multilingual Speech-to-Text AI</div>
        <div class="header-desc">
            Proprietary Speech Recognition for <b>Gujarati (ગુજરાતી)</b>, <b>Hindi (हिंदी)</b> & <b>English</b>.<br>
            Optimized with Acoustic Normalization, Beam Search & Gujarati Conjunct Correction.
        </div>
        """
    )

    with gr.Row():
        with gr.Column(scale=1):
            audio_input = gr.Audio(
                sources=["microphone", "upload"],
                type="filepath",
                label="🎤 Speak via Microphone or Upload File",
            )
            lang_dropdown = gr.Dropdown(
                choices=["Gujarati (ગુજરાતી)", "Hindi (हिंदी)", "English", "Auto Detect"],
                value="Gujarati (ગુજરાતી)",
                label="🌐 Target Language",
            )
            transcribe_btn = gr.Button("🚀 Transcribe Speech", variant="primary", size="lg")

        with gr.Column(scale=1):
            output_text = gr.Textbox(
                label="📝 Transcription Result",
                placeholder="તમારું લખાણ અહીં દેખાશે...",
                lines=5,
            )
            status_text = gr.Textbox(
                label="ℹ️ System Diagnostics",
                interactive=False,
                lines=1,
            )

    transcribe_btn.click(
        fn=transcribe,
        inputs=[audio_input, lang_dropdown],
        outputs=[output_text, status_text],
    )

if __name__ == "__main__":
    demo.launch()
