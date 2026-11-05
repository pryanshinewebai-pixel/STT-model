"""
Inference Engine — Runs Multilingual STT Model Inference on Audio Inputs.

Supports both single audio file processing and batch processing.
Optimized for CPU/MPS inference on macOS.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import soundfile as sf
import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor

logger = logging.getLogger(__name__)

DEFAULT_MODEL_DIR = Path(__file__).resolve().parent.parent / "models" / "finetuned_whisper"


class STTInference:
    """Multilingual Speech-to-Text inference engine.

    Loads the fine-tuned Whisper model checkpoint and provides methods
    for transcribing audio files or raw audio arrays.
    """

    LANG_CODES = {
        "en": "en",
        "english": "en",
        "hi": "hi",
        "hindi": "hi",
        "gu": "gu",
        "gujarati": "gu",
    }

    def __init__(
        self,
        model_path: str | Path = DEFAULT_MODEL_DIR,
        device: str = "cpu",
    ) -> None:
        """Initialize the inference engine.

        Args:
            model_path: Path to the trained model directory containing safetensors.
            device: Device for inference ('cpu', 'mps', 'cuda').
        """
        self.model_path = Path(model_path)
        if device == "auto":
            if torch.backends.mps.is_available():
                self.device = torch.device("mps")
            elif torch.cuda.is_available():
                self.device = torch.device("cuda")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)

        logger.info(f"Initializing STTInference on {self.device} from {self.model_path}...")
        self._load_model()

    def _load_model(self) -> None:
        """Load fine-tuned model and processor."""
        if not self.model_path.exists():
            remote_model = os.environ.get("WHISPER_REMOTE_MODEL", "openai/whisper-tiny")
            logger.warning(f"Local model path not found at {self.model_path}. Loading '{remote_model}'...")
            self.model = WhisperForConditionalGeneration.from_pretrained(remote_model).to(self.device)
            self.processor = WhisperProcessor.from_pretrained(remote_model)
            self.model.eval()
            logger.info("✅ Whisper model and processor loaded successfully!")
            return

        self.model = WhisperForConditionalGeneration.from_pretrained(str(self.model_path)).to(self.device)
        self.model.eval()

        # Load processor (processor configs are stored in model_dir or default whisper-tiny)
        try:
            self.processor = WhisperProcessor.from_pretrained(str(self.model_path))
        except Exception:
            logger.info("Using standard whisper-small processor mapping...")
            self.processor = WhisperProcessor.from_pretrained("openai/whisper-small")

        logger.info("✅ Fine-tuned STT model and processor loaded successfully!")

    def transcribe_array(
        self,
        waveform: np.ndarray,
        sample_rate: int = 16000,
        language: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Transcribe a numpy audio waveform array.

        Args:
            waveform: 1D numpy array of float32 audio samples.
            sample_rate: Sample rate of audio (default: 16000).
            language: Optional language hint ('en', 'hi', 'gu').

        Returns:
            Dictionary with 'text', 'language', 'duration'.
        """
        if waveform.ndim > 1:
            waveform = waveform.mean(axis=1)

        waveform = waveform.astype(np.float32)
        duration = round(len(waveform) / sample_rate, 2)

        # Silence / Muted Microphone Detection (prevents Whisper hallucination on empty input)
        max_amp = float(np.max(np.abs(waveform))) if len(waveform) > 0 else 0.0
        if max_amp < 0.005 or duration < 0.3:
            logger.warning(f"Audio is silent (max amplitude: {max_amp:.4f}). Skipping Whisper hallucination.")
            return {
                "text": "⚠️ (Aawaz record nahi hui / Mic muted tha. Kripya thoda tez bolein ya browser mic permission check karein!)",
                "duration": duration,
                "language": "none",
            }

        # Resample to 16kHz if needed
        if sample_rate != 16000:
            import librosa
            waveform = librosa.resample(waveform, orig_sr=sample_rate, target_sr=16000)
            sample_rate = 16000

        # Normalize language code if provided
        lang_code = self.LANG_CODES.get(language.lower() if language else "", None)
        forced_decoder_ids = None
        if lang_code:
            forced_decoder_ids = self.processor.get_decoder_prompt_ids(language=lang_code, task="transcribe")

        inputs = self.processor(waveform, sampling_rate=sample_rate, return_tensors="pt").input_features.to(self.device)

        with torch.no_grad():
            if forced_decoder_ids is not None:
                predicted_ids = self.model.generate(inputs, forced_decoder_ids=forced_decoder_ids)
            else:
                predicted_ids = self.model.generate(inputs)

        transcription = self.processor.batch_decode(predicted_ids, skip_special_tokens=True)[0].strip()

        return {
            "text": transcription,
            "duration": duration,
            "language": lang_code or "auto",
        }

    def transcribe_file(
        self,
        audio_path: str | Path,
        language: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Transcribe a single audio file.

        Args:
            audio_path: Path to the audio file (.wav, .mp3, .flac).
            language: Optional target language code ('en', 'hi', 'gu').

        Returns:
            Dictionary with transcription result.
        """
        p = Path(audio_path)
        if not p.exists():
            raise FileNotFoundError(f"Audio file not found: {p}")

        try:
            audio, sr = sf.read(str(p), dtype="float32")
        except Exception:
            import subprocess, tempfile
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_out:
                tmp_out_path = tmp_out.name
            try:
                subprocess.run(
                    ["/usr/bin/afconvert", "-f", "WAVE", "-d", "LEI16@16000", "-c", "1", str(p), tmp_out_path],
                    check=True,
                    capture_output=True
                )
                audio, sr = sf.read(tmp_out_path, dtype="float32")
            finally:
                if os.path.exists(tmp_out_path):
                    try:
                        os.unlink(tmp_out_path)
                    except Exception:
                        pass

        result = self.transcribe_array(audio, sample_rate=sr, language=language)
        result["file_name"] = p.name
        result["audio_path"] = str(p.resolve())
        return result

    def transcribe_batch(
        self,
        audio_paths: List[str | Path],
        language: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Transcribe a batch of audio files."""
        return [self.transcribe_file(p, language=language) for p in audio_paths]
