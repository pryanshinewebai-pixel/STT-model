---
title: Multilingual Indic STT
emoji: 🎙️
colorFrom: indigo
colorTo: blue
sdk: gradio
sdk_version: 4.44.0
app_file: app.py
pinned: false
license: mit
---

# 🎙️ Multilingual Speech-to-Text AI (Gujarati, Hindi, English)

A fine-tuned speech recognition web application supporting **Gujarati (ગુજરાતી)**, **Hindi (हिंदी)**, and **English**, optimized with acoustic normalization, silence trimming, beam search, and Gujarati conjunct correction.

### Features
- 🎤 Live microphone input and audio file upload.
- 🌐 Manual language selection or automatic language routing.
- ⚡ In-memory Gujarati orthography conjunct normalizer (fixes common ASR spelling variations).
- 🔇 Smart silence trimming to eliminate trailing hallucinations.
