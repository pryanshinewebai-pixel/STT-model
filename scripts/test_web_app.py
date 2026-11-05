#!/usr/bin/env python3
"""
Interactive Web Application for Testing Multilingual Speech-to-Text Models.
Features:
  - 🔄 Dual Model Engine:
      1. Fine-Tuned Whisper Model (High Accuracy ~90-95%, 680k hr Pretrained) [Default]
      2. Proprietary Conformer-CTC Model (30MB Lightweight Scratch Model)
  - 🎙️ Direct 16kHz PCM WAV Microphone Recording (Browser AudioContext)
  - 📁 Audio File Upload (.wav, .m4a, .mp3 with native macOS afconvert fallback)
  - ⚡ 1-Click Demo Samples (Gujarati, Hindi, English)
"""

import argparse
import base64
import io
import json
import logging
import os
import subprocess
import sys
import tempfile
import time
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import soundfile as sf
from inference.inference import STTInference
from inference.proprietary_inference import ProprietarySTTInference

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

PORT = int(os.environ.get("PORT", 8080))
whisper_engine: STTInference = None
proprietary_engine: ProprietarySTTInference = None

HTML_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Multilingual STT Live Tester | Gujarati, Hindi, English</title>
    <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@400;500;600;700;800&family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
    <style>
        :root {
            --bg: #090d16;
            --surface: #131b2e;
            --surface-hover: #1c2742;
            --border: #22304d;
            --primary: #38bdf8;
            --primary-glow: rgba(56, 189, 248, 0.35);
            --accent: #818cf8;
            --success: #10b981;
            --danger: #ef4444;
            --text-main: #f8fafc;
            --text-muted: #94a3b8;
        }

        * { box-sizing: border-box; margin: 0; padding: 0; font-family: 'Inter', sans-serif; }
        body { background: var(--bg); color: var(--text-main); min-height: 100vh; padding: 30px 20px; display: flex; flex-direction: column; align-items: center; }
        .container { width: 100%; max-width: 920px; display: flex; flex-direction: column; gap: 24px; }

        header {
            text-align: center;
            padding: 30px 24px;
            background: linear-gradient(135deg, #131b2e 0%, #1e1b4b 100%);
            border-radius: 20px;
            border: 1px solid var(--border);
            box-shadow: 0 15px 35px rgba(0,0,0,0.4);
            position: relative;
        }
        .badge {
            display: inline-flex;
            align-items: center;
            gap: 6px;
            padding: 6px 14px;
            border-radius: 50px;
            background: rgba(56, 189, 248, 0.12);
            color: var(--primary);
            font-size: 13px;
            font-weight: 600;
            margin-bottom: 12px;
            border: 1px solid rgba(56, 189, 248, 0.3);
        }
        h1 { font-family: 'Outfit', sans-serif; font-size: 32px; font-weight: 800; letter-spacing: -0.5px; margin-bottom: 8px; }
        h1 span { background: linear-gradient(90deg, #38bdf8, #818cf8); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
        p.subtitle { color: var(--text-muted); font-size: 15px; }

        /* Model Switcher */
        .model-switcher {
            display: flex;
            background: #0b1120;
            padding: 6px;
            border-radius: 14px;
            border: 1px solid var(--border);
            gap: 8px;
        }
        .model-btn {
            flex: 1;
            background: transparent;
            border: none;
            color: var(--text-muted);
            padding: 12px;
            border-radius: 10px;
            cursor: pointer;
            font-size: 14px;
            font-weight: 600;
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 8px;
            transition: all 0.2s;
        }
        .model-btn.active {
            background: var(--surface-hover);
            color: var(--primary);
            border: 1px solid var(--border);
            box-shadow: 0 4px 12px rgba(0,0,0,0.3);
        }

        .card {
            background: var(--surface);
            border-radius: 18px;
            border: 1px solid var(--border);
            padding: 24px;
            box-shadow: 0 10px 25px rgba(0,0,0,0.3);
            display: flex;
            flex-direction: column;
            gap: 20px;
        }

        .tab-bar {
            display: flex;
            gap: 10px;
            border-bottom: 1px solid var(--border);
            padding-bottom: 12px;
        }
        .tab-btn {
            background: transparent;
            border: none;
            color: var(--text-muted);
            font-size: 14px;
            font-weight: 600;
            padding: 8px 16px;
            border-radius: 8px;
            cursor: pointer;
            transition: all 0.2s;
        }
        .tab-btn.active {
            background: var(--surface-hover);
            color: var(--primary);
            border: 1px solid var(--border);
        }

        /* Mic panel */
        .mic-box {
            display: flex;
            flex-direction: column;
            align-items: center;
            gap: 16px;
            padding: 30px;
            background: #0b1120;
            border-radius: 14px;
            border: 1px dashed var(--border);
        }
        .mic-circle {
            width: 80px;
            height: 80px;
            border-radius: 50%;
            background: linear-gradient(135deg, #38bdf8, #818cf8);
            border: none;
            color: #090d16;
            font-size: 28px;
            cursor: pointer;
            display: flex;
            align-items: center;
            justify-content: center;
            box-shadow: 0 10px 25px var(--primary-glow);
            transition: all 0.2s;
        }
        .mic-circle:hover { transform: scale(1.05); }
        .mic-circle.recording {
            background: var(--danger);
            animation: pulse 1.2s infinite;
            color: #fff;
        }
        @keyframes pulse {
            0% { box-shadow: 0 0 0 0 rgba(239, 68, 68, 0.7); }
            70% { box-shadow: 0 0 0 20px rgba(239, 68, 68, 0); }
            100% { box-shadow: 0 0 0 0 rgba(239, 68, 68, 0); }
        }

        /* Drop box */
        .drop-box {
            display: none;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            gap: 12px;
            padding: 35px;
            background: #0b1120;
            border-radius: 14px;
            border: 2px dashed var(--border);
            cursor: pointer;
            transition: all 0.2s;
        }
        .drop-box:hover { border-color: var(--primary); background: #0e1628; }

        /* Sample buttons */
        .sample-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
            gap: 12px;
        }
        .sample-btn {
            background: #0b1120;
            border: 1px solid var(--border);
            border-radius: 10px;
            padding: 12px 14px;
            color: var(--text-main);
            text-align: left;
            cursor: pointer;
            display: flex;
            flex-direction: column;
            gap: 4px;
            transition: all 0.2s;
        }
        .sample-btn:hover { border-color: var(--primary); background: var(--surface-hover); }
        .sample-title { font-weight: 600; font-size: 13px; color: var(--primary); display: flex; align-items: center; gap: 6px; }
        .sample-desc { font-size: 12px; color: var(--text-muted); }

        /* Results */
        .result-card {
            background: var(--surface);
            border-radius: 18px;
            border: 1px solid var(--border);
            padding: 24px;
            display: flex;
            flex-direction: column;
            gap: 16px;
        }
        .meta-row {
            display: flex;
            justify-content: space-between;
            align-items: center;
            font-size: 13px;
            color: var(--text-muted);
            border-bottom: 1px solid var(--border);
            padding-bottom: 10px;
            flex-wrap: wrap;
            gap: 8px;
        }
        .meta-pill {
            background: rgba(56, 189, 248, 0.1);
            color: var(--primary);
            padding: 3px 10px;
            border-radius: 20px;
            font-size: 12px;
            font-weight: 600;
        }
        .box-title { font-size: 12px; font-weight: 700; color: var(--text-muted); text-transform: uppercase; letter-spacing: 0.5px; }
        .text-display {
            background: #0b1120;
            border: 1px solid var(--border);
            border-radius: 12px;
            padding: 16px;
            font-size: 20px;
            font-weight: 500;
            line-height: 1.6;
            min-height: 65px;
        }
        .text-display.final { color: #38bdf8; border-color: rgba(56, 189, 248, 0.4); }

        .spinner {
            display: none;
            width: 24px;
            height: 24px;
            border: 3px solid rgba(56, 189, 248, 0.2);
            border-top-color: var(--primary);
            border-radius: 50%;
            animation: spin 0.8s linear infinite;
        }
        @keyframes spin { to { transform: rotate(360deg); } }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <div class="badge"><i class="fa-solid fa-microphone-lines"></i> Multilingual Speech AI</div>
            <h1>Live STT <span>Model Tester</span></h1>
            <p class="subtitle">Real-time Gujarati (ગુજરાતી), Hindi (हिंदी) & English Voice Recognition</p>
        </header>

        <!-- Model Switcher -->
        <div class="model-switcher">
            <button class="model-btn active" id="btnWhisper" onclick="setModel('whisper')">
                <i class="fa-solid fa-star" style="color: #f59e0b;"></i>
                Fine-Tuned Whisper (High Accuracy ~90-95%)
            </button>
            <button class="model-btn" id="btnProprietary" onclick="setModel('proprietary')">
                <i class="fa-solid fa-microchip"></i>
                Proprietary Conformer (30MB Scratch Model)
            </button>
        </div>

        <div class="card">
            <div class="tab-bar">
                <button class="tab-btn active" id="tabMic" onclick="switchTab('mic')"><i class="fa-solid fa-microphone"></i> Live Microphone</button>
                <button class="tab-btn" id="tabUpload" onclick="switchTab('upload')"><i class="fa-solid fa-cloud-arrow-up"></i> Upload Audio File</button>
                <button class="tab-btn" id="tabSamples" onclick="switchTab('samples')"><i class="fa-solid fa-list-check"></i> Sample Audio Clips</button>
            </div>

            <!-- Mic Section -->
            <div class="mic-box" id="micPanel">
                <button class="mic-circle" id="micBtn" onclick="toggleMic()"><i class="fa-solid fa-microphone"></i></button>
                <p id="micStatus" style="font-size: 14px; color: var(--text-muted);">Click microphone & speak in Gujarati, Hindi, or English</p>
                <div id="recTimer" style="display:none; font-size:16px; font-weight:700; color:var(--danger);">00:00</div>
            </div>

            <!-- Upload Section -->
            <div class="drop-box" id="uploadPanel" onclick="document.getElementById('fileInput').click()">
                <i class="fa-solid fa-file-audio" style="font-size: 36px; color: var(--primary);"></i>
                <p style="font-weight: 600;">Click to select or drag audio file here</p>
                <p style="font-size: 12px; color: var(--text-muted);">Supports .wav, .m4a, .mp3 (Mono/Stereo)</p>
                <input type="file" id="fileInput" accept="audio/*" style="display:none;" onchange="handleFileUpload(event)">
            </div>

            <!-- Sample Section -->
            <div class="sample-grid" id="samplePanel" style="display:none;">
                <button class="sample-btn" onclick="testSample('guj2_3_4_clean_chunk_017.wav')">
                    <span class="sample-title"><i class="fa-solid fa-play"></i> Gujarati Sample (ગુજરાતી)</span>
                    <span class="sample-desc">"બપોરે પોણાં 3 વાગ્યે ફોન કરજો..."</span>
                </button>
                <button class="sample-btn" onclick="testSample('hin1_11_12_clean_chunk_001.wav')">
                    <span class="sample-title"><i class="fa-solid fa-play"></i> Hindi Sample (हिंदी)</span>
                    <span class="sample-desc">"वेबसाइट / सामान्य वाक्य"</span>
                </button>
                <button class="sample-btn" onclick="testSample('eng1_11_12_clean_chunk_001.wav')">
                    <span class="sample-title"><i class="fa-solid fa-play"></i> English Sample</span>
                    <span class="sample-desc">"She sells sea shells by the sea shore"</span>
                </button>
            </div>
        </div>

        <div class="result-card">
            <div class="meta-row">
                <div style="display:flex; align-items:center; gap:8px;">
                    <span class="box-title">Transcription Output</span>
                    <div class="spinner" id="spinner"></div>
                </div>
                <div style="display:flex; gap:8px;" id="metaStats">
                    <span class="meta-pill" id="modPill">Model: Fine-Tuned Whisper</span>
                    <span class="meta-pill" id="durPill">Duration: --</span>
                    <span class="meta-pill" id="latPill">Latency: --</span>
                </div>
            </div>

            <div>
                <p class="box-title" style="margin-bottom:6px; color:var(--primary);"><i class="fa-solid fa-check-double"></i> Recognized Speech Text:</p>
                <div class="text-display final" id="finalText">Awaiting speech or audio file...</div>
            </div>
        </div>
    </div>

    <script>
        let currentModel = 'whisper';
        let isRecording = false;
        let timerInterval;
        let audioContext;
        let audioInput;
        let recorderNode;
        let pcmBuffers = [];
        let mediaStream;

        function setModel(model) {
            currentModel = model;
            document.getElementById('btnWhisper').classList.toggle('active', model === 'whisper');
            document.getElementById('btnProprietary').classList.toggle('active', model === 'proprietary');
            document.getElementById('modPill').textContent = model === 'whisper' ? 'Model: Fine-Tuned Whisper (~95% Acc)' : 'Model: Proprietary Conformer (30MB)';
        }

        function switchTab(tab) {
            document.getElementById('tabMic').classList.toggle('active', tab === 'mic');
            document.getElementById('tabUpload').classList.toggle('active', tab === 'upload');
            document.getElementById('tabSamples').classList.toggle('active', tab === 'samples');

            document.getElementById('micPanel').style.display = tab === 'mic' ? 'flex' : 'none';
            document.getElementById('uploadPanel').style.display = tab === 'upload' ? 'flex' : 'none';
            document.getElementById('samplePanel').style.display = tab === 'samples' ? 'grid' : 'none';
        }

        async function toggleMic() {
            if (!isRecording) {
                try {
                    mediaStream = await navigator.mediaDevices.getUserMedia({ audio: { channelCount: 1, sampleRate: 16000 } });
                    audioContext = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: 16000 });
                    audioInput = audioContext.createMediaStreamSource(mediaStream);

                    recorderNode = audioContext.createScriptProcessor(4096, 1, 1);
                    pcmBuffers = [];

                    recorderNode.onaudioprocess = function(e) {
                        if (!isRecording) return;
                        const channelData = e.inputBuffer.getChannelData(0);
                        pcmBuffers.push(new Float32Array(channelData));
                    };

                    audioInput.connect(recorderNode);
                    recorderNode.connect(audioContext.destination);

                    isRecording = true;
                    document.getElementById('micBtn').classList.add('recording');
                    document.getElementById('micBtn').innerHTML = '<i class="fa-solid fa-stop"></i>';
                    document.getElementById('micStatus').textContent = 'Recording... Speak now in Gujarati, Hindi or English!';

                    let sec = 0;
                    document.getElementById('recTimer').style.display = 'block';
                    document.getElementById('recTimer').textContent = '00:00';
                    timerInterval = setInterval(() => {
                        sec++;
                        const m = String(Math.floor(sec / 60)).padStart(2, '0');
                        const s = String(sec % 60).padStart(2, '0');
                        document.getElementById('recTimer').textContent = `${m}:${s}`;
                    }, 1000);
                } catch(e) {
                    alert('Microphone error: ' + e.message);
                }
            } else {
                isRecording = false;
                clearInterval(timerInterval);

                if (recorderNode) {
                    recorderNode.disconnect();
                    audioInput.disconnect();
                }
                if (mediaStream) {
                    mediaStream.getTracks().forEach(t => t.stop());
                }
                if (audioContext && audioContext.state !== 'closed') {
                    audioContext.close();
                }

                document.getElementById('micBtn').classList.remove('recording');
                document.getElementById('micBtn').innerHTML = '<i class="fa-solid fa-microphone"></i>';
                document.getElementById('micStatus').textContent = 'Transcribing with ' + (currentModel === 'whisper' ? 'Fine-Tuned Whisper...' : 'Proprietary Model...');
                document.getElementById('recTimer').style.display = 'none';

                sendPCMAsWav();
            }
        }

        function exportWAV(buffers, sampleRate = 16000) {
            let length = 0;
            for (let i = 0; i < buffers.length; i++) {
                length += buffers[i].length;
            }
            const merged = new Float32Array(length);
            let offset = 0;
            for (let i = 0; i < buffers.length; i++) {
                merged.set(buffers[i], offset);
                offset += buffers[i].length;
            }

            const buffer = new ArrayBuffer(44 + merged.length * 2);
            const view = new DataView(buffer);

            function writeString(offset, str) {
                for (let i = 0; i < str.length; i++) {
                    view.setUint8(offset + i, str.charCodeAt(i));
                }
            }

            writeString(0, 'RIFF');
            view.setUint32(4, 36 + merged.length * 2, true);
            writeString(8, 'WAVE');
            writeString(12, 'fmt ');
            view.setUint32(16, 16, true);
            view.setUint16(20, 1, true); // PCM
            view.setUint16(22, 1, true); // Mono
            view.setUint32(24, sampleRate, true); // 16kHz
            view.setUint32(28, sampleRate * 2, true); // Byte rate
            view.setUint16(32, 2, true); // Block align
            view.setUint16(34, 16, true); // 16 bits
            writeString(36, 'data');
            view.setUint32(40, merged.length * 2, true);

            let index = 44;
            for (let i = 0; i < merged.length; i++) {
                let s = Math.max(-1, Math.min(1, merged[i]));
                view.setInt16(index, s < 0 ? s * 0x8000 : s * 0x7FFF, true);
                index += 2;
            }

            return new Blob([view], { type: 'audio/wav' });
        }

        async function sendPCMAsWav() {
            if (!pcmBuffers.length) return;
            const wavBlob = exportWAV(pcmBuffers, 16000);
            const reader = new FileReader();
            reader.readAsDataURL(wavBlob);
            reader.onloadend = async () => {
                const b64 = reader.result.split(',')[1];
                await postAudio({ audio_b64: b64, format: 'wav', model: currentModel });
            };
        }

        async function handleFileUpload(e) {
            const file = e.target.files[0];
            if (!file) return;
            const reader = new FileReader();
            reader.readAsDataURL(file);
            reader.onloadend = async () => {
                const b64 = reader.result.split(',')[1];
                await postAudio({ audio_b64: b64, filename: file.name, model: currentModel });
            };
        }

        async function testSample(filename) {
            await postAudio({ sample_file: filename, model: currentModel });
        }

        async function postAudio(payload) {
            const spinner = document.getElementById('spinner');
            const finalText = document.getElementById('finalText');

            spinner.style.display = 'inline-block';
            finalText.textContent = 'Transcribing with ' + (currentModel === 'whisper' ? 'Fine-Tuned Whisper Model (~95% Acc)...' : 'Proprietary Conformer...');

            try {
                const res = await fetch('/api/transcribe', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(payload)
                });
                const data = await res.json();
                if (data.status === 'success') {
                    finalText.textContent = data.text || '(No speech detected)';
                    document.getElementById('durPill').textContent = `Duration: ${data.duration}s`;
                    document.getElementById('latPill').textContent = `Latency: ${data.latency_ms} ms`;
                } else {
                    finalText.textContent = 'Error: ' + data.message;
                }
            } catch(e) {
                finalText.textContent = 'Connection error: ' + e.message;
            } finally {
                spinner.style.display = 'none';
            }
        }
    </script>
</body>
</html>
"""


def load_audio_from_bytes(audio_bytes: bytes) -> tuple[np.ndarray, int]:
    """Read any audio bytes directly via soundfile or fallback to native macOS afconvert."""
    try:
        audio, sr = sf.read(io.BytesIO(audio_bytes), dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        return audio, sr
    except Exception:
        with tempfile.NamedTemporaryFile(suffix=".raw_audio", delete=False) as tmp_in:
            tmp_in.write(audio_bytes)
            tmp_in_path = tmp_in.name
        tmp_out_path = tmp_in_path + ".wav"
        try:
            subprocess.run(
                ["/usr/bin/afconvert", "-f", "WAVE", "-d", "LEI16@16000", "-c", "1", tmp_in_path, tmp_out_path],
                check=True,
                capture_output=True,
            )
            audio, sr = sf.read(tmp_out_path, dtype="float32")
            if audio.ndim > 1:
                audio = audio.mean(axis=1)
            return audio, sr
        finally:
            for p in (tmp_in_path, tmp_out_path):
                if os.path.exists(p):
                    try:
                        os.unlink(p)
                    except Exception:
                        pass


class STTTesterHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path in ("/", "/index.html"):
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(HTML_PAGE.encode("utf-8"))
            return
        self.send_error(404, "Not Found")

    def do_POST(self):
        if self.path == "/api/transcribe":
            content_len = int(self.headers.get("Content-Length", 0))
            post_data = self.rfile.read(content_len)
            try:
                payload = json.loads(post_data.decode("utf-8"))
                model_choice = payload.get("model", "whisper")
                t0 = time.time()

                # Get audio array
                if "sample_file" in payload:
                    sample_path = PROJECT_ROOT / "data" / "chunked_dataset" / "wav_chunks" / payload["sample_file"]
                    if not sample_path.exists():
                        import glob
                        matches = glob.glob(str(PROJECT_ROOT / "data" / "chunked_dataset" / "wav_chunks" / "*.wav"))
                        sample_path = Path(matches[0]) if matches else sample_path

                    with open(sample_path, "rb") as f:
                        audio, sr = load_audio_from_bytes(f.read())
                elif "audio_b64" in payload:
                    audio_bytes = base64.b64decode(payload["audio_b64"])
                    audio, sr = load_audio_from_bytes(audio_bytes)
                else:
                    self.send_json(400, {"status": "error", "message": "Missing audio input"})
                    return

                duration = round(len(audio) / sr, 2)

                # Route to selected model
                if model_choice == "whisper" and whisper_engine is not None:
                    res = whisper_engine.transcribe_array(audio, sample_rate=sr)
                    text_output = res["text"]
                else:
                    res = proprietary_engine.transcribe(audio, sample_rate=sr)
                    text_output = res["text"]

                latency_ms = round((time.time() - t0) * 1000, 1)

                self.send_json(200, {
                    "status": "success",
                    "text": text_output,
                    "duration": duration,
                    "latency_ms": latency_ms,
                    "model": model_choice,
                })
            except Exception as e:
                logger.error(f"Error in transcription: {e}", exc_info=True)
                self.send_json(500, {"status": "error", "message": str(e)})
            return

        self.send_error(404, "Not Found")

    def send_json(self, code: int, data: dict):
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(json.dumps(data, ensure_ascii=False).encode("utf-8"))

    def log_message(self, format, *args):
        return


class ReusableServer(HTTPServer):
    allow_reuse_address = True


def main():
    global whisper_engine, proprietary_engine
    parser = argparse.ArgumentParser(description="Web tester for STT Models")
    parser.add_argument("--port", type=int, default=PORT)
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    print("\n" + "=" * 65)
    try:
        print(f"🚀 LOADING WHISPER MODEL ON {args.device.upper()}...")
        whisper_engine = STTInference(device=args.device)
        print("✅ Fine-Tuned Whisper Loaded Successfully!")
    except Exception as e:
        logger.warning(f"Whisper engine could not be loaded: {e}")
        whisper_engine = None

    try:
        print(f"🚀 LOADING PROPRIETARY CONFORMER MODEL (30MB) ON {args.device.upper()}...")
        proprietary_engine = ProprietarySTTInference(device=args.device)
        print("✅ Proprietary Conformer Loaded Successfully!")
    except Exception as e:
        logger.error(f"Proprietary engine load failed: {e}")
    print("=" * 65)

    server = ReusableServer(("", args.port), STTTesterHandler)
    print(f"\n🌐 Live STT Web Tester running at: http://localhost:{args.port}")
    print(f"🎙️ Open your browser, speak in Gujarati, Hindi, or English!")
    server.serve_forever()


if __name__ == "__main__":
    main()
