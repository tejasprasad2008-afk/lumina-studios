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

## 🚀 Quick Start

### Prerequisites
- Python 3.11+
- ffmpeg — `brew install ffmpeg` (Mac) / `sudo apt install ffmpeg` (Linux)

### Install & Run
```bash
git clone https://github.com/PierrunoYT/Kokoro-TTS-Local
cd Kokoro-TTS-Local
chmod +x start.sh
./start.sh
```
Open **http://localhost:8000** 🎉

### Windows
```bash
git clone https://github.com/PierrunoYT/Kokoro-TTS-Local
cd Kokoro-TTS-Local
pip install -r requirements.txt
python server.py
```

## 🎭 RVC Voice Models
1. Download `.pth` models from [weights.gg](https://weights.gg) or [Hugging Face](https://huggingface.co)
2. Upload via the sidebar in the app
3. Optionally add a `.index` file for better similarity

Recommended starter model: search **"CBTNarrator RVC"** on Hugging Face.

## 🎙️ Podcast Generator
Converts any text/article into a two-host podcast conversation.

Requires a **free** Groq API key:
1. Go to [console.groq.com](https://console.groq.com) — no credit card needed
2. Create an API key
3. Add to `.env`: `GROQ_API_KEY=your_key_here`

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