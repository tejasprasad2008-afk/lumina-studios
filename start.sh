#!/bin/bash
echo "🎙️ Starting Lumina TTS Studio..."

if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3.11+ required. Download at python.org"
    exit 1
fi

if ! command -v ffmpeg &> /dev/null; then
    echo "❌ ffmpeg required."
    echo "   Mac: brew install ffmpeg"
    echo "   Linux: sudo apt install ffmpeg"
    echo "   Windows: https://ffmpeg.org/download.html"
    exit 1
fi

if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
fi

source venv/bin/activate 2>/dev/null || . venv/Scripts/activate 2>/dev/null

echo "📦 Installing dependencies..."
pip install -r requirements.txt -q

if [ ! -f ".env" ]; then
    cp .env.example .env
    echo "⚙️  .env created — add GROQ_API_KEY for Podcast Generator (optional)"
fi

mkdir -p outputs kokoro_temp rvc_models sfx foley/generated

echo ""
echo "✅ Lumina is running at http://localhost:8000"
echo "   Press Ctrl+C to stop"
echo ""
python server.py
