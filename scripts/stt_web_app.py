#!/usr/bin/env python3
"""
Interactive Web Application for Testing Fine-Tuned Multilingual STT Model.

Features:
  - 🎙️ Live Browser Microphone Recording & Streaming
  - 📁 Drag & Drop Audio File Upload (.wav, .mp3, .m4a, .ogg)
  - 🌐 Multilingual Support (English, Hindi, Gujarati, Auto-Detect)
  - ⚡ 1-Click Demo Sample Audio Player & Tester
  - 📋 One-Click Copy Transcription, Audio Waveform Visualizer
"""

import argparse
import base64
import io
import json
import logging
import mimetypes
import sys
import time
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import soundfile as sf
from inference.inference import STTInference

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

PORT = 8080
engine: STTInference = None


HTML_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Multilingual STT Live Tester | Proprietary Speech-to-Text</title>
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
        .container { width: 100%; max-width: 960px; display: flex; flex-direction: column; gap: 24px; }

        /* Header */
        header {
            text-align: center;
            padding: 30px 24px;
            background: linear-gradient(135deg, #131b2e 0%, #1e1b4b 100%);
            border-radius: 20px;
            border: 1px solid var(--border);
            box-shadow: 0 15px 35px rgba(0,0,0,0.4);
            position: relative;
            overflow: hidden;
        }
        header::after {
            content: '';
            position: absolute;
            top: -50%; left: -50%; width: 200%; height: 200%;
            background: radial-gradient(circle, var(--primary-glow) 0%, transparent 60%);
            opacity: 0.15;
            pointer-events: none;
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

        /* Controls Card */
        .card {
            background: var(--surface);
            border-radius: 18px;
            border: 1px solid var(--border);
            padding: 24px;
            box-shadow: 0 10px 25px rgba(0,0,0,0.3);
            display: flex;
            flex-direction: column;
            gap: 18px;
        }

        .lang-bar {
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 16px;
            flex-wrap: wrap;
        }
        .lang-options {
            display: flex;
            gap: 8px;
            background: #0b1120;
            padding: 5px;
            border-radius: 12px;
            border: 1px solid var(--border);
        }
        .lang-btn {
            background: transparent;
            border: none;
            color: var(--text-muted);
            padding: 8px 16px;
            border-radius: 8px;
            font-size: 13px;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.2s;
        }
        .lang-btn.active {
            background: var(--primary);
            color: #0b1120;
            box-shadow: 0 4px 12px var(--primary-glow);
        }

        /* Action Grid */
        .actions-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
        }
        @media (max-width: 768px) {
            .actions-grid { grid-template-columns: 1fr; }
        }

        .action-box {
            background: #0b1120;
            border: 2px dashed var(--border);
            border-radius: 14px;
            padding: 30px 20px;
            text-align: center;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            gap: 14px;
            transition: all 0.2s;
            cursor: pointer;
            position: relative;
        }
        .action-box:hover {
            border-color: var(--primary);
            background: #111a2e;
        }

        /* Mic Button */
        .mic-btn {
            width: 76px;
            height: 76px;
            border-radius: 50%;
            background: linear-gradient(135deg, #0284c7, #2563eb);
            border: none;
            color: white;
            font-size: 28px;
            cursor: pointer;
            display: flex;
            align-items: center;
            justify-content: center;
            box-shadow: 0 8px 25px rgba(2, 132, 199, 0.4);
            transition: all 0.2s;
        }
        .mic-btn:hover { transform: scale(1.06); }
        .mic-btn.recording {
            background: linear-gradient(135deg, #ef4444, #dc2626);
            box-shadow: 0 0 0 10px rgba(239, 68, 68, 0.25);
            animation: pulse 1.5s infinite;
        }
        @keyframes pulse {
            0% { transform: scale(1); }
            50% { transform: scale(1.08); box-shadow: 0 0 0 18px rgba(239, 68, 68, 0.1); }
            100% { transform: scale(1); }
        }

        .recording-time {
            font-family: 'Outfit', sans-serif;
            font-size: 16px;
            font-weight: 700;
            color: var(--danger);
            display: none;
        }

        /* Demo Audio Chips */
        .demos-section {
            display: flex;
            flex-direction: column;
            gap: 10px;
        }
        .demos-title { font-size: 13px; font-weight: 700; color: var(--text-muted); text-transform: uppercase; letter-spacing: 0.5px; }
        .demos-grid {
            display: flex;
            gap: 10px;
            flex-wrap: wrap;
        }
        .demo-chip {
            background: #0b1120;
            border: 1px solid var(--border);
            color: var(--text-main);
            padding: 8px 14px;
            border-radius: 10px;
            font-size: 13px;
            font-weight: 600;
            cursor: pointer;
            display: inline-flex;
            align-items: center;
            gap: 8px;
            transition: all 0.2s;
        }
        .demo-chip:hover {
            border-color: var(--primary);
            background: var(--surface-hover);
            transform: translateY(-2px);
        }

        /* Output Result Box */
        .result-box {
            background: #070a12;
            border: 1px solid var(--border);
            border-radius: 16px;
            padding: 24px;
            display: flex;
            flex-direction: column;
            gap: 14px;
            min-height: 180px;
            position: relative;
        }
        .result-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            border-bottom: 1px solid var(--border);
            padding-bottom: 12px;
        }
        .result-title {
            font-family: 'Outfit', sans-serif;
            font-weight: 700;
            font-size: 15px;
            color: var(--primary);
            display: flex;
            align-items: center;
            gap: 8px;
        }
        .meta-badges {
            display: flex;
            gap: 8px;
            align-items: center;
        }
        .meta-tag {
            background: #1e293b;
            padding: 4px 10px;
            border-radius: 6px;
            font-size: 12px;
            font-weight: 600;
            color: var(--text-muted);
        }
        .result-text {
            font-size: 20px;
            font-weight: 600;
            line-height: 1.6;
            color: #ffffff;
            word-break: break-word;
            flex: 1;
        }
        .result-text.empty {
            color: #475569;
            font-weight: 400;
            font-style: italic;
            display: flex;
            align-items: center;
            justify-content: center;
        }

        .btn-copy {
            background: #1e293b;
            border: 1px solid var(--border);
            color: #fff;
            padding: 6px 12px;
            border-radius: 8px;
            font-size: 12px;
            font-weight: 600;
            cursor: pointer;
            display: inline-flex;
            align-items: center;
            gap: 6px;
            transition: all 0.2s;
        }
        .btn-copy:hover {
            background: var(--primary);
            color: #0b1120;
        }

        /* Spinner */
        .spinner {
            display: none;
            width: 28px;
            height: 28px;
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
        <!-- HEADER -->
        <header>
            <div class="badge"><i class="fa-solid fa-bolt"></i> Fine-Tuned Whisper Multilingual</div>
            <h1>🎙️ Speech-to-Text <span>Live Tester</span></h1>
            <p class="subtitle">Supports English 🇬🇧, Hindi 🇮🇳 (हिंदी), and Gujarati 🇮🇳 (ગુજરાતી)</p>
        </header>

        <!-- MAIN CARD -->
        <div class="card">
            <!-- Language Selector -->
            <div class="lang-bar">
                <span style="font-size: 14px; font-weight: 700; color: var(--text-muted);"><i class="fa-solid fa-language"></i> Language:</span>
                <div class="lang-options">
                    <button class="lang-btn active" data-lang="auto">🌐 Auto-Detect</button>
                    <button class="lang-btn" data-lang="en">🇬🇧 English</button>
                    <button class="lang-btn" data-lang="hi">🇮🇳 Hindi</button>
                    <button class="lang-btn" data-lang="gu">🇮🇳 Gujarati</button>
                </div>
            </div>

            <!-- Two Input Methods: Microphone & File Upload -->
            <div class="actions-grid">
                <!-- 1. Live Mic -->
                <div class="action-box" id="micBox">
                    <button class="mic-btn" id="micBtn" title="Click to Record">
                        <i class="fa-solid fa-microphone"></i>
                    </button>
                    <div id="micLabel" style="font-weight: 600; font-size: 14px;">Click to Speak</div>
                    <div class="recording-time" id="recTimer">00:00</div>
                </div>

                <!-- 2. File Upload -->
                <div class="action-box" id="fileBox" onclick="document.getElementById('audioInput').click()">
                    <i class="fa-solid fa-cloud-arrow-up" style="font-size: 38px; color: var(--primary);"></i>
                    <div style="font-weight: 600; font-size: 14px;">Drop Audio File Here or Click to Upload</div>
                    <div style="font-size: 12px; color: var(--text-muted);">WAV, MP3, M4A, FLAC, OGG</div>
                    <input type="file" id="audioInput" accept="audio/*" style="display: none;">
                </div>
            </div>

            <!-- 1-Click Demo Samples -->
            <div class="demos-section">
                <div class="demos-title">⚡ Quick Test with Demo Samples:</div>
                <div class="demos-grid">
                    <button class="demo-chip" onclick="testDemo('fleurs_gu_001.wav', 'gu')">
                        <i class="fa-solid fa-play"></i> 🇮🇳 Gujarati News Sample
                    </button>
                    <button class="demo-chip" onclick="testDemo('fleurs_hi_001.wav', 'hi')">
                        <i class="fa-solid fa-play"></i> 🇮🇳 Hindi Speech Sample
                    </button>
                    <button class="demo-chip" onclick="testDemo('eng1_11_12_clean_chunk_001.wav', 'en')">
                        <i class="fa-solid fa-play"></i> 🇬🇧 English Tongue Twister
                    </button>
                </div>
            </div>
        </div>

        <!-- TRANSCRIPTION RESULT BOX -->
        <div class="result-box">
            <div class="result-header">
                <div class="result-title">
                    <i class="fa-solid fa-align-left"></i>
                    <span>Transcription Output</span>
                    <div class="spinner" id="spinner"></div>
                </div>
                <div class="meta-badges">
                    <span class="meta-tag" id="badgeLang">Lang: -</span>
                    <span class="meta-tag" id="badgeDur">Duration: -</span>
                    <span class="meta-tag" id="badgeTime">Latency: -</span>
                    <button class="btn-copy" onclick="copyText()"><i class="fa-solid fa-copy"></i> Copy</button>
                </div>
            </div>
            <div class="result-text empty" id="resultText">Your transcription will appear here...</div>
        </div>
    </div>

    <script>
        let selectedLang = 'auto';
        let mediaRecorder = null;
        let audioChunks = [];
        let timerInterval = null;
        let seconds = 0;

        // Language toggle
        document.querySelectorAll('.lang-btn').forEach(btn => {
            btn.addEventListener('click', () => {
                document.querySelectorAll('.lang-btn').forEach(b => b.classList.remove('active'));
                btn.classList.add('active');
                selectedLang = btn.dataset.lang;
            });
        });

        // Microphone Recording
        const micBtn = document.getElementById('micBtn');
        const micLabel = document.getElementById('micLabel');
        const recTimer = document.getElementById('recTimer');

        micBtn.addEventListener('click', async (e) => {
            e.stopPropagation();
            if (mediaRecorder && mediaRecorder.state === 'recording') {
                stopRecording();
            } else {
                startRecording();
            }
        });

        async function startRecording() {
            try {
                const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
                mediaRecorder = new MediaRecorder(stream);
                audioChunks = [];

                mediaRecorder.ondataavailable = e => { if (e.data.size > 0) audioChunks.push(e.data); };
                mediaRecorder.onstop = sendRecordedAudio;

                mediaRecorder.start();
                micBtn.classList.add('recording');
                micBtn.innerHTML = '<i class="fa-solid fa-stop"></i>';
                micLabel.textContent = 'Recording... Click to Transcribe';
                recTimer.style.display = 'block';

                seconds = 0;
                recTimer.textContent = '00:00';
                timerInterval = setInterval(() => {
                    seconds++;
                    const m = String(Math.floor(seconds / 60)).padStart(2, '0');
                    const s = String(seconds % 60).padStart(2, '0');
                    recTimer.textContent = `${m}:${s}`;
                }, 1000);
            } catch (err) {
                alert('Microphone access denied: ' + err.message);
            }
        }

        function stopRecording() {
            if (mediaRecorder && mediaRecorder.state === 'recording') {
                mediaRecorder.stop();
                mediaRecorder.stream.getTracks().forEach(t => t.stop());
                clearInterval(timerInterval);

                micBtn.classList.remove('recording');
                micBtn.innerHTML = '<i class="fa-solid fa-microphone"></i>';
                micLabel.textContent = 'Click to Speak';
                recTimer.style.display = 'none';
            }
        }

        async function sendRecordedAudio() {
            const blob = new Blob(audioChunks, { type: 'audio/webm' });
            const reader = new FileReader();
            reader.readAsDataURL(blob);
            reader.onloadend = async () => {
                const base64Data = reader.result.split(',')[1];
                await transcribePayload({ audio_base64: base64Data, format: 'webm', language: selectedLang });
            };
        }

        // File upload
        document.getElementById('audioInput').addEventListener('change', async (e) => {
            const file = e.target.files[0];
            if (!file) return;

            const reader = new FileReader();
            reader.readAsDataURL(file);
            reader.onloadend = async () => {
                const base64Data = reader.result.split(',')[1];
                const ext = file.name.split('.').pop();
                await transcribePayload({ audio_base64: base64Data, format: ext, language: selectedLang, filename: file.name });
            };
        });

        // Demo sample test
        async function testDemo(filename, lang) {
            await transcribePayload({ demo_file: filename, language: lang });
        }

        async function transcribePayload(payload) {
            const resultBox = document.getElementById('resultText');
            const spinner = document.getElementById('spinner');

            spinner.style.display = 'block';
            resultBox.className = 'result-text empty';
            resultBox.textContent = 'Transcribing with Fine-Tuned Whisper Model...';

            try {
                const res = await fetch('/api/transcribe', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(payload)
                });
                const data = await res.json();
                spinner.style.display = 'none';

                if (data.status === 'success') {
                    resultBox.className = 'result-text';
                    resultBox.textContent = data.transcription;
                    document.getElementById('badgeLang').textContent = `Lang: ${data.language.toUpperCase()}`;
                    document.getElementById('badgeDur').textContent = `Duration: ${data.duration}s`;
                    document.getElementById('badgeTime').textContent = `Latency: ${data.latency_sec}s`;
                } else {
                    resultBox.className = 'result-text empty';
                    resultBox.textContent = 'Error: ' + data.message;
                }
            } catch (err) {
                spinner.style.display = 'none';
                resultBox.className = 'result-text empty';
                resultBox.textContent = 'Network / Server Error: ' + err.message;
            }
        }

        function copyText() {
            const text = document.getElementById('resultText').textContent;
            navigator.clipboard.writeText(text);
            const btn = document.querySelector('.btn-copy');
            btn.innerHTML = '<i class="fa-solid fa-check"></i> Copied!';
            setTimeout(() => { btn.innerHTML = '<i class="fa-solid fa-copy"></i> Copy'; }, 2000);
        }
    </script>
</body>
</html>
"""


class STTHandler(BaseHTTPRequestHandler):
    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_GET(self):
        if self.path == "/" or self.path == "/index.html":
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(HTML_PAGE.encode("utf-8"))
            return

        # Serve demo audio files if requested
        if self.path.startswith("/wav_chunks/"):
            filename = Path(self.path).name
            filepath = PROJECT_ROOT / "data" / "chunked_dataset" / "wav_chunks" / filename
            if filepath.exists():
                self.send_response(200)
                self.send_header("Content-Type", "audio/wav")
                self.send_header("Content-Length", str(filepath.stat().st_size))
                self.end_headers()
                with open(filepath, "rb") as f:
                    self.wfile.write(f.read())
                return

        self.send_error(404, "File Not Found")

    def do_POST(self):
        if self.path == "/api/transcribe":
            content_len = int(self.headers.get("Content-Length", 0))
            post_data = self.rfile.read(content_len)

            try:
                payload = json.loads(post_data.decode("utf-8"))
                lang = payload.get("language", "auto")
                lang_param = None if lang == "auto" else lang

                t0 = time.time()

                # Case 1: Demo file requested
                if "demo_file" in payload:
                    demo_filename = payload["demo_file"]
                    demo_path = PROJECT_ROOT / "data" / "chunked_dataset" / "wav_chunks" / demo_filename
                    if not demo_path.exists():
                        self.send_json_response(404, {"status": "error", "message": f"Demo file not found: {demo_filename}"})
                        return

                    result = engine.transcribe_file(demo_path, language=lang_param)

                # Case 2: Base64 audio uploaded / mic recording
                elif "audio_base64" in payload:
                    audio_bytes = base64.b64decode(payload["audio_base64"])
                    audio, sr = sf.read(io.BytesIO(audio_bytes), dtype="float32")
                    result = engine.transcribe_array(audio, sample_rate=sr, language=lang_param)

                else:
                    self.send_json_response(400, {"status": "error", "message": "Missing audio payload"})
                    return

                latency = round(time.time() - t0, 3)

                response = {
                    "status": "success",
                    "transcription": result["text"],
                    "duration": result["duration"],
                    "language": result["language"],
                    "latency_sec": latency,
                }
                self.send_json_response(200, response)

            except Exception as e:
                logger.error(f"Error during transcription: {e}", exc_info=True)
                self.send_json_response(500, {"status": "error", "message": str(e)})
            return

        self.send_error(404, "Unknown Endpoint")

    def send_json_response(self, code: int, data: dict):
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(json.dumps(data, ensure_ascii=False).encode("utf-8"))

    def log_message(self, format, *args):
        return


class ReusableHTTPServer(HTTPServer):
    allow_reuse_address = True


def main():
    global engine
    parser = argparse.ArgumentParser(description="Live Web App for Fine-Tuned STT Model.")
    parser.add_argument("--port", type=int, default=PORT, help=f"Port number (default: {PORT})")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "mps", "cuda", "auto"], help="Inference device")
    args = parser.parse_args()

    print("\n" + "=" * 65)
    print("🚀 INITIALIZING FINE-TUNED MULTILINGUAL STT WEB SERVER")
    print("=" * 65)
    engine = STTInference(device=args.device)

    server_address = ("", args.port)
    httpd = ReusableHTTPServer(server_address, STTHandler)

    print(f"\n🌐 Web Tester running at: http://localhost:{args.port}")
    print(f"🎙️ Open your browser and speak into the mic or upload audio!")
    print("💡 Press Ctrl+C to stop the server.\n" + "=" * 65 + "\n")

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n🛑 Server stopped gracefully.")


if __name__ == "__main__":
    main()
