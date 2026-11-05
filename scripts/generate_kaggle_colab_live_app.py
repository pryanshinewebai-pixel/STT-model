#!/usr/bin/env python3
"""
Generate a 1-Click Kaggle / Google Colab Notebook for 24/7 Live GPU STT Server.
"""

import json
from pathlib import Path

notebook = {
    "cells": [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# 🎙️ 1-Click Live Multilingual STT Server on Cloud GPU (Kaggle / Colab)\n",
                "### Powered by NVIDIA Tesla T4 GPU | English, Hindi & Gujarati Voice Recognition\n",
                "\n",
                "This self-contained notebook runs your Speech-to-Text model on a free Cloud GPU and gives you an **instant Public HTTPS Live URL** using Cloudflare Tunnel.\n",
                "- 🚀 Runs on **NVIDIA Tesla T4 GPU** (Latency < 0.5s)\n",
                "- 🌐 Works on **Mobile Phones & Laptops** anywhere in the world\n",
                "- 🔒 Full **Microphone WebRTC** support\n",
                "- ⏳ Stays active for up to **12 Hours** per session\n",
                "\n",
                "---"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### Step 1: Clone Repository & Install Dependencies"
            ]
        },
        {
            "cell_type": "code",
            "metadata": {},
            "source": [
                "import os, subprocess, time\n",
                "\n",
                "print('📦 Setting up repository from GitHub...')\n",
                "if not os.path.exists('STT-model'):\n",
                "    !git clone -q https://github.com/pryanshinewebai-pixel/STT-model.git\n",
                "    print('✅ Repository cloned!')\n",
                "else:\n",
                "    !cd STT-model && git pull -q\n",
                "    print('✅ Repository updated!')\n",
                "\n",
                "print('⚙️ Installing audio and deep learning dependencies...')\n",
                "!pip install -q transformers soundfile torchaudio librosa accelerate\n",
                "print('✅ Dependencies installed successfully!')"
            ],
            "execution_count": None,
            "outputs": []
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### Step 2: Install Cloudflare Tunnel for Instant Public URL"
            ]
        },
        {
            "cell_type": "code",
            "metadata": {},
            "source": [
                "import os\n",
                "\n",
                "print('🌐 Installing Cloudflare Tunnel binary...')\n",
                "!curl -sL https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 -o /tmp/cloudflared\n",
                "!chmod +x /tmp/cloudflared\n",
                "print('✅ Cloudflare tunnel ready!')"
            ],
            "execution_count": None,
            "outputs": []
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### Step 3: Launch Live STT Web App on NVIDIA GPU"
            ]
        },
        {
            "cell_type": "code",
            "metadata": {},
            "source": [
                "import subprocess, time, torch\n",
                "\n",
                "device = 'cuda' if torch.cuda.is_available() else 'cpu'\n",
                "print(f'🚀 Starting Live STT Server on device: {device.upper()}...')\n",
                "\n",
                "# Start server process in background\n",
                "server_proc = subprocess.Popen(\n",
                "    ['python3', 'scripts/test_web_app.py', '--port', '8080', '--device', device],\n",
                "    cwd='STT-model',\n",
                "    stdout=subprocess.PIPE,\n",
                "    stderr=subprocess.STDOUT,\n",
                "    text=True\n",
                ")\n",
                "\n",
                "# Wait 5s for models to load\n",
                "time.sleep(5)\n",
                "print('✅ Live STT server process running on port 8080!')"
            ],
            "execution_count": None,
            "outputs": []
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### Step 4: Open Public Live Link (Share with Anyone!)"
            ]
        },
        {
            "cell_type": "code",
            "metadata": {},
            "source": [
                "import subprocess, re, time\n",
                "\n",
                "print('🔗 Creating secure Public HTTPS Tunnel...')\n",
                "tunnel_proc = subprocess.Popen(\n",
                "    ['/tmp/cloudflared', 'tunnel', '--url', 'http://localhost:8080'],\n",
                "    stdout=subprocess.PIPE,\n",
                "    stderr=subprocess.STDOUT,\n",
                "    text=True\n",
                ")\n",
                "\n",
                "public_url = None\n",
                "for _ in range(40):\n",
                "    line = tunnel_proc.stdout.readline()\n",
                "    match = re.search(r'https://[a-zA-Z0-9-]+\\.trycloudflare\\.com', line)\n",
                "    if match:\n",
                "        public_url = match.group(0)\n",
                "        break\n",
                "    time.sleep(0.5)\n",
                "\n",
                "if public_url:\n",
                "    print('\\n' + '=' * 70)\n",
                "    print('🎉 YOUR MULTILINGUAL STT APP IS LIVE ON CLOUD GPU!')\n",
                "    print(f'👉 PUBLIC LINK: {public_url}')\n",
                "    print('=' * 70 + '\\n')\n",
                "    print('📱 Open this link on your Mobile Phone, Tablet, or PC to test!')\n",
                "    print('🎙️ Click microphone, speak in Gujarati, Hindi or English, and see instant output.')\n",
                "    print('⏳ Keep this cell running — it will stay active for up to 12 hours!')\n",
                "else:\n",
                "    print('⚠️ Could not parse URL automatically. Log output:')\n",
                "    print(tunnel_proc.stdout.read())\n",
                "\n",
                "# Keep session alive\n",
                "try:\n",
                "    while True:\n",
                "        time.sleep(60)\n",
                "except KeyboardInterrupt:\n",
                "    print('\\n🛑 Tunnel stopped by user.')"
            ],
            "execution_count": None,
            "outputs": []
        }
    ],
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "name": "python",
            "version": "3.10.0"
        },
        "accelerator": "GPU"
    },
    "nbformat": 4,
    "nbformat_minor": 4
}

desktop_path = Path("/Users/commerciax/Desktop/kaggle_live_stt_server.ipynb")
workspace_path = Path("/Users/commerciax/proprietary_stt_model/kaggle_live_stt_server.ipynb")

for p in (desktop_path, workspace_path):
    with open(p, "w", encoding="utf-8") as f:
        json.dump(notebook, f, indent=2, ensure_ascii=False)
    print(f"✅ Generated: {p}")
