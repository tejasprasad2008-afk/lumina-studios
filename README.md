# 🎙️ Lumina TTS Studio

A fully local, privacy-first Text-to-Speech studio powered by Kokoro TTS 
with RVC voice conversion, emotion-aware narration, and a cinematic web UI.

![Python](https://img.shields.io/badge/Python-3.11-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-latest-green)
![License](https://img.shields.io/badge/License-MIT-yellow)
![Local](https://img.shields.io/badge/Runs-100%25%20Local-brightgreen)

## ✨ Features

| Feature | Details |
|---|---|
| 🎙️ 54+ Voices | Kokoro TTS — fully local, no API |
| 🎭 RVC Conversion | Transform output into any voice model |
| 🧠 Emotion-Aware | Auto breath & pause injection |
| 📦 Low RAM Mode | Chunked processing for long texts |
| 🌡️ Thermal Management | Smart GPU cooldown between chunks |
| 🎵 Acoustic Humanizer | EQ, speed & normalization |
| 🎧 Waveform Player | Wavesurfer.js with speed control |
| 🔒 Security Hardened | CSRF, rate limiting, XSS headers |
| 🌀 Holographic UI | WebGL orb loading animation |

## 🚀 Installation Guide

### Step 1 — Prerequisites

**Python 3.11**
- Download from [python.org](https://python.org/downloads)
- During install on Windows, check ✅ "Add Python to PATH"
- Verify: `python --version` should show `3.11.x`

**ffmpeg**
- Mac: `brew install ffmpeg` (need [Homebrew](https://brew.sh) first)
- Linux: `sudo apt install ffmpeg`
- Windows: Download from [ffmpeg.org](https://ffmpeg.org/download.html), add to PATH
- Verify: `ffmpeg -version`

**Git**
- Download from [git-scm.com](https://git-scm.com)
- Verify: `git --version`

---

### Step 2 — Clone the repository
```bash
git clone https://github.com/tejasprasad2008-afk/lumina-studios.git
cd lumina-studios
```

---

### Step 3 — Install Kokoro TTS

Lumina uses Kokoro as its TTS engine. Install it first:
```bash
pip install kokoro>=0.9.4 soundfile
```

**Mac/Linux — also install espeak (for phonemization):**
```bash
# Mac
brew install espeak-ng

# Linux
sudo apt install espeak-ng
```

**Windows — install espeak:**
Download from [espeak-ng.org](https://github.com/espeak-ng/espeak-ng/releases) and install.

> **Note:** On first run, Kokoro will automatically download the model 
> weights (~300MB) from Hugging Face. This is a one-time download.

---

### Step 4 — Start Lumina

**Mac/Linux:**
```bash
chmod +x start.sh
./start.sh
```

**Windows:**
```bash
pip install -r requirements.txt
python server.py
```

The first run will:
1. Create a virtual environment
2. Install all dependencies
3. Download Kokoro model weights (~300MB, one time only)
4. Start the server

---

### Step 5 — Open the app

Once you see this in your terminal:
```
✅ Lumina is running at http://localhost:8000
```

Open your browser and go to:
**[http://localhost:8000](http://localhost:8000)** 🎉

---

### Step 6 — (Optional) RVC Voice Models

To use voice conversion for more human-sounding output:
1. Go to [weights.gg](https://weights.gg) or [Hugging Face](https://huggingface.co)
2. Search `CBTNarrator RVC` for a great narrator voice
3. Download the `.pth` file (and `.index` if available)
4. In the app sidebar → Voice Conversion → upload your `.pth` file
5. Enable the RVC toggle and generate

---

### Step 7 — (Optional) Podcast Generator

To use the two-host podcast feature:
1. Go to [console.groq.com](https://console.groq.com) — free, no credit card
2. Create an API key
3. Open the `.env` file in the project folder
4. Replace `your_groq_api_key_here` with your actual key
5. Restart the server
6. Navigate to the Podcast tab

---

### ❓ Troubleshooting

| Problem | Fix |
|---|---|
| `ffmpeg not found` | Install ffmpeg and restart terminal |
| `kokoro not found` | Run `pip install kokoro` |
| `Port 8000 in use` | Run `kill -9 $(lsof -t -i:8000)` then retry |
| Model download stuck | Check internet connection, retry |
| RVC conversion skipping | Check `.pth` file is valid, re-upload |
| Mac running hot | Enable Low RAM Mode in sidebar |

## 🔒 Security
> ⚠️ Lumina is designed for **local use only**.  
> Never expose port 8000 to the public internet.

Built-in protections: rate limiting, CSRF tokens, CORS lockdown, 
prompt injection filtering, path traversal jailing, magic-byte 
file validation, XSS/CSP headers, redacted logging, 10MB request limit.

## 📋 Full Requirements
See `requirements.txt`. Key dependencies:
- `kokoro` — TTS engine
- `rvc-python` — voice conversion  
- `pydub` + `ffmpeg` — audio processing
- `fastapi` + `uvicorn` — web server
- `pedalboard` — audio effects
- `slowapi` — rate limiting
- `groq` — podcast script generation (optional)

## 🤝 Contributing
PRs welcome. Please test on Python 3.11 before submitting.

## 📄 License
MIT — free to use, modify, and distribute.

---
Built with 🎙️ by [Tejas](https://github.com/tejasprasad2008-afk)