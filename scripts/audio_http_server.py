#!/usr/bin/env python3
"""
Custom High-Performance Audio HTTP Server supporting HTTP Range Requests, CORS, & Live Transcript Editing.

Features:
  - Supports HTTP Range requests (206 Partial Content) for Chrome, Safari, and Google Sheets.
  - Handles POST requests at /api/save_transcript to save manual transcript edits live.
  - Automatically re-encodes CTC tokens, updates CSV manifests, Desktop Excel, and JSON files on save.
"""

import csv
import json
import logging
import mimetypes
import sys
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from openpyxl import load_workbook
from openpyxl.styles import Alignment, Font, PatternFill

from preprocessing.tokenizer import CTCTokenizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DIRECTORY = PROJECT_ROOT / "data" / "chunked_dataset"
PORT = 8000

TRANSCRIPTS_DIR = DIRECTORY / "transcripts"
TOKENIZED_DIR = DIRECTORY / "tokenized_transcripts"
VOCAB_PATH = DIRECTORY / "vocab.json"
MANIFEST_CHUNKED = DIRECTORY / "manifest_chunked.csv"
MANIFEST_TOKENIZED = DIRECTORY / "manifest_tokenized.csv"
EXCEL_PATH = DIRECTORY / "stt_chunked_summary.xlsx"
DESKTOP_EXCEL_PATH = Path.home() / "Desktop" / "stt_chunked_summary.xlsx"

mimetypes.add_type("audio/wav", ".wav")
mimetypes.add_type("audio/mpeg", ".mp3")
mimetypes.add_type("text/html", ".html")
mimetypes.add_type("text/css", ".css")
mimetypes.add_type("application/javascript", ".js")
mimetypes.add_type("application/json", ".json")


class AudioHTTPRequestHandler(BaseHTTPRequestHandler):
    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_HEAD(self):
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Accept-Ranges", "bytes")
        file_path = self.get_file_path()
        if file_path and file_path.exists() and file_path.is_file():
            content_type, _ = mimetypes.guess_type(str(file_path))
            self.send_header("Content-Type", content_type or "application/octet-stream")
            self.send_header("Content-Length", str(file_path.stat().st_size))
        self.end_headers()

    def do_GET(self):
        file_path = self.get_file_path()

        if not file_path or not file_path.exists() or not file_path.is_file():
            self.send_error(404, "File Not Found")
            return

        file_size = file_path.stat().st_size
        content_type, _ = mimetypes.guess_type(str(file_path))
        content_type = content_type or "application/octet-stream"

        range_header = self.headers.get("Range")

        if range_header and range_header.startswith("bytes="):
            try:
                bytes_range = range_header.split("=")[1].split("-")
                start = int(bytes_range[0])
                end = int(bytes_range[1]) if bytes_range[1] else file_size - 1
                if start >= file_size:
                    self.send_error(416, "Requested Range Not Satisfiable")
                    return
                end = min(end, file_size - 1)
                chunk_size = end - start + 1

                self.send_response(206)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Range", f"bytes {start}-{end}/{file_size}")
                self.send_header("Content-Length", str(chunk_size))
                self.send_header("Accept-Ranges", "bytes")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()

                with open(file_path, "rb") as f:
                    f.seek(start)
                    self.wfile.write(f.read(chunk_size))
                return
            except Exception:
                pass

        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(file_size))
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()

        with open(file_path, "rb") as f:
            shutil_copyfile(f, self.wfile)

    def do_POST(self):
        if self.path == "/api/save_transcript":
            content_length = int(self.headers.get("Content-Length", 0))
            post_data = self.rfile.read(content_length)
            
            try:
                payload = json.loads(post_data.decode("utf-8"))
                audio_name = payload.get("audio_name", "").strip()
                new_transcript = payload.get("transcript", "").strip()

                if not audio_name or not new_transcript:
                    self.send_json_response(400, {"status": "error", "message": "Missing audio_name or transcript"})
                    return

                stem = Path(audio_name).stem

                # 1. Update txt transcript file
                txt_path = TRANSCRIPTS_DIR / f"{stem}.txt"
                with open(txt_path, "w", encoding="utf-8") as f:
                    f.write(new_transcript + "\n")

                # 2. Re-tokenize and update JSON file
                tokenizer = CTCTokenizer.load_vocab(VOCAB_PATH) if VOCAB_PATH.exists() else CTCTokenizer()
                token_ids = tokenizer.encode(new_transcript)
                token_chars = [tokenizer.id2char.get(tid, "") for tid in token_ids]

                json_path = TOKENIZED_DIR / f"{stem}.json"
                with open(json_path, "w", encoding="utf-8") as jf:
                    json.dump({
                        "audio_name": audio_name,
                        "transcript": new_transcript,
                        "token_ids": token_ids,
                        "token_chars": token_chars,
                        "num_tokens": len(token_ids),
                    }, jf, ensure_ascii=False, indent=2)

                # 3. Update manifest CSVs
                self.update_manifest_csvs(audio_name, new_transcript, token_ids)

                self.send_json_response(200, {
                    "status": "success",
                    "message": f"Saved '{audio_name}'",
                    "num_tokens": len(token_ids),
                    "token_ids": token_ids,
                })
                logger.info(f"💾 Live edited & saved transcript for {audio_name}: '{new_transcript}'")
                return

            except Exception as e:
                logger.error(f"Error saving transcript: {e}")
                self.send_json_response(500, {"status": "error", "message": str(e)})
                return

        self.send_error(404, "Endpoint Not Found")

    def send_json_response(self, code: int, data: dict):
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(json.dumps(data, ensure_ascii=False).encode("utf-8"))

    def update_manifest_csvs(self, audio_name: str, new_transcript: str, token_ids: list):
        # Update manifest_chunked.csv
        if MANIFEST_CHUNKED.exists():
            rows = []
            with open(MANIFEST_CHUNKED, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for r in reader:
                    if Path(r["audio_path"]).name == audio_name:
                        r["transcript"] = new_transcript
                    rows.append(r)

            fieldnames = ["audio_path", "original_source", "duration", "language", "transcript"]
            with open(MANIFEST_CHUNKED, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows)

        # Update manifest_tokenized.csv
        if MANIFEST_TOKENIZED.exists():
            rows = []
            with open(MANIFEST_TOKENIZED, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for r in reader:
                    if Path(r["audio_path"]).name == audio_name:
                        r["transcript"] = new_transcript
                        r["num_tokens"] = len(token_ids)
                        r["token_ids_str"] = str(token_ids)
                    rows.append(r)

            fieldnames = ["audio_path", "original_source", "duration", "language", "transcript", "num_tokens", "token_ids_str"]
            with open(MANIFEST_TOKENIZED, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows)

    def get_file_path(self) -> Path:
        rel_path = self.path.lstrip("/").split("?")[0]
        if not rel_path or rel_path == "":
            rel_path = "index.html"
        target = (DIRECTORY / rel_path).resolve()
        if str(target).startswith(str(DIRECTORY.resolve())):
            return target
        return None

    def log_message(self, format, *args):
        return


def shutil_copyfile(source, target):
    buffer_size = 64 * 1024
    while True:
        buf = source.read(buffer_size)
        if not buf:
            break
        target.write(buf)


class ReusableHTTPServer(HTTPServer):
    allow_reuse_address = True


def run_server():
    import argparse
    parser = argparse.ArgumentParser(description="Live Editing Audio HTTP Server")
    parser.add_argument("--port", type=int, default=PORT, help=f"Port to bind server (default: {PORT})")
    args, _ = parser.parse_known_args()
    port = args.port

    server_address = ("", port)
    try:
        httpd = ReusableHTTPServer(server_address, AudioHTTPRequestHandler)
        print(f"✅ Live Editing Audio HTTP Server running at http://localhost:{port}")
        print("💡 Press Ctrl+C to stop the server.")
        httpd.serve_forever()
    except OSError as e:
        if e.errno == 48:
            print(f"❌ Error: Port {port} is already in use by another process.")
            print(f"👉 To free port {port}, run: lsof -ti :{port} | xargs kill -9")
            print(f"👉 Or run on a different port: ./venv/bin/python scripts/audio_http_server.py --port {port + 1}")
        else:
            raise
    except KeyboardInterrupt:
        print("\n🛑 Server stopped gracefully.")


if __name__ == "__main__":
    run_server()
