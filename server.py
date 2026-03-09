import os
import sys
from pathlib import Path
from typing import List, Optional
from datetime import datetime
import torch
import numpy as np
import soundfile as sf
from pydub import AudioSegment
from fastapi import FastAPI, HTTPException, Request, BackgroundTasks, UploadFile, File
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import uvicorn
import logging
from fastapi.responses import StreamingResponse
import json
import asyncio
import shutil
from rvc_python.infer import RVCInference
import httpx
import re
from groq import Groq
from dotenv import load_dotenv
from pedalboard import Pedalboard, Reverb
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from starlette.middleware.base import BaseHTTPMiddleware
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import subprocess
import secrets

load_dotenv()

# Add the current directory to sys.path to import local modules
sys.path.append(os.path.abspath("."))

from models import (
    list_available_voices, build_model,
    EnhancedKPipeline
)

# Constants
DEFAULT_OUTPUT_DIR = Path("outputs")
DEFAULT_SAMPLE_RATE = 24000
MAX_TEXT_LENGTH = 5000

# Global device and pipelines
device = 'cuda' if torch.cuda.is_available() else 'cpu'
pipelines = {}
rvc = RVCInference(device=device)

LANG_MAP = {
    "af_": "a", "am_": "a",
    "bf_": "b", "bm_": "b",
    "jf_": "j", "jm_": "j",
    "zf_": "z", "zm_": "z",
    "ef_": "e", "em_": "e",
    "ff_": "f",
    "hf_": "h", "hm_": "h",
    "if_": "i", "im_": "i",
    "pf_": "p", "pm_": "p",
}

app = FastAPI(title="Lumina TTS API")

# 1 — Rate Limiting
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# 8 — CORS lockdown
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8000", "http://127.0.0.1:8000"],
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "X-CSRF-Token"],
    allow_credentials=False,
)

# 6 — XSS protection (Security Headers)
class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        logger.info("🔍 Middleware (Security) hit: %s %s", request.method, request.url.path)
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; "
            "script-src 'self' 'unsafe-inline' https://cdnjs.cloudflare.com https://unpkg.com; "
            "style-src 'self' 'unsafe-inline' https://fonts.googleapis.com; "
            "font-src https://fonts.gstatic.com; "
            "media-src 'self' blob:; "
            "connect-src 'self';"
        )
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["Permissions-Policy"] = "microphone=(), camera=(), geolocation=()"
        return response

app.add_middleware(SecurityHeadersMiddleware)

# 9 — Request size limit (DoS protection)
class RequestSizeLimitMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        logger.info("🔍 Middleware (SizeLimit) hit: %s %s", request.method, request.url.path)
        if request.method == "POST":
            content_length = request.headers.get("content-length")
            if content_length and int(content_length) > 10 * 1024 * 1024:
                return JSONResponse(status_code=413, content={"error": "Request too large — 10MB max"})
        return await call_next(request)

app.add_middleware(RequestSizeLimitMiddleware)

@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    logger.warning("🔹 HTTP Error %d: %s", exc.status_code, exc.detail)
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": exc.detail}
    )

@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    logger.error("❌ Unhandled error: %s", str(exc), exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"error": f"Internal server error: {str(exc)}"}
    )

# Logging and SSE setup
log_queue = asyncio.Queue()

class QueueHandler(logging.Handler):
    def emit(self, record):
        msg = self.format(record)
        try:
            loop = asyncio.get_running_loop()
            loop.create_task(log_queue.put(msg))
        except RuntimeError:
            pass

logger = logging.getLogger("kokoro")
logger.setLevel(logging.INFO)
logger.handlers = []

# Terminal Output
stream_handler = logging.StreamHandler(sys.stdout)
stream_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
logger.addHandler(stream_handler)

# Frontend Console Output
q_handler = QueueHandler()
q_handler.setFormatter(logging.Formatter('%(levelname)s: %(message)s'))
logger.addHandler(q_handler)

@app.get("/logs")
async def stream_logs():
    async def generate():
        while True:
            msg = await log_queue.get()
            yield f"data: {msg}\n\n"
    return StreamingResponse(generate(), media_type="text/event-stream")

# Ensure output directory exists
DEFAULT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
TEMP_DIR = Path("./kokoro_temp/")
TEMP_DIR.mkdir(parents=True, exist_ok=True)
RVC_MODELS_DIR = Path("./rvc_models")
RVC_MODELS_DIR.mkdir(parents=True, exist_ok=True)

# Mount outputs for direct access
app.mount("/outputs", StaticFiles(directory="outputs"), name="outputs")

# 2 — SQL/Command Injection protection (Filename Sanitization)
def sanitize_filename(name: str) -> str:
    # remove path traversal, null bytes, special chars
    name = name.replace('\x00', '')
    name = re.sub(r'[^\w\s\-.]', '_', name)
    name = re.sub(r'\.\.+', '.', name)  # collapse multiple dots
    name = name.strip('. ')
    if not name:
        raise HTTPException(status_code=400, detail="Invalid filename")
    return name[:100]  # max 100 chars

# 3 — Path Traversal protection (Safe Path Jailing)
def safe_path(base_dir: Path, filename: str) -> Path:
    base = base_dir.resolve()
    target = (base / filename).resolve()
    if not str(target).startswith(str(base)):
        raise HTTPException(status_code=403, detail="Access denied")
    return target

# 4 — Prompt Injection protection (LLM inputs)
def sanitize_for_llm(text: str) -> str:
    injection_patterns = [
        r'ignore\s+(all\s+)?(previous|prior|above)\s+instructions',
        r'you\s+are\s+now',
        r'new\s+(system\s+)?instruction',
        r'disregard\s+(all\s+)?',
        r'forget\s+(everything|all|your)',
        r'act\s+as\s+(a\s+|an\s+)?(?!podcast|narrator)',
        r'jailbreak',
        r'dan\s+mode',
        r'\[(system|assistant|user|inst|instruction)\]',
        r'<\s*/?\s*(system|instruction|prompt)\s*>',
        r'###\s*instruction',
        r'human:\s',
        r'assistant:\s',
    ]
    for pattern in injection_patterns:
        if re.search(pattern, text, re.IGNORECASE):
            raise HTTPException(status_code=400, detail="Invalid input detected")
    return text

def sanitize_input(text: str, max_words: int = 5000) -> str:
    text = text.replace('\x00', '').strip()
    if len(text.split()) > max_words:
        raise HTTPException(status_code=400, detail=f"Exceeds {max_words} word limit")
    if len(text) > 100000:
        raise HTTPException(status_code=400, detail="Input too large")
    if len(text.strip()) < 3:
        raise HTTPException(status_code=400, detail="Input too short")
    return text

# 5 — File Upload security (Malicious File Upload protection)
async def validate_upload(file: UploadFile, allowed_extensions: set, max_mb: int):
    ext = Path(file.filename).suffix.lower()
    if ext not in allowed_extensions:
        raise HTTPException(status_code=400, detail=f"File type not allowed. Allowed: {allowed_extensions}")
    
    contents = await file.read()
    max_bytes = max_mb * 1024 * 1024
    if len(contents) > max_bytes:
        raise HTTPException(status_code=400, detail=f"File exceeds {max_mb}MB limit")
    
    # magic bytes check — verify file is actually what it claims
    # .pth files are zip format internally
    if ext == '.pth':
        # Check for zip signature (PK\x03\x04) or pickle signature (\x80\x02)
        if not (contents[:4] == b'PK\x03\x04' or contents[:2] == b'\x80\x02'):
            raise HTTPException(status_code=400, detail="Invalid model file format")
    
    await file.seek(0)
    return sanitize_filename(file.filename), contents

# 7 — CSRF protection
csrf_tokens = set()

@app.get("/csrf_token")
async def get_csrf_token():
    token = secrets.token_hex(32)
    csrf_tokens.add(token)
    return {"token": token}

def validate_csrf(request: Request):
    token = request.headers.get("X-CSRF-Token")
    logger.info("CSRF check — token received: %s", token[:8] if token else "NONE")
    if not token or token not in csrf_tokens:
        logger.error("CSRF validation failed — token not in valid set")
        raise HTTPException(status_code=403, detail="Invalid CSRF token")
    csrf_tokens.discard(token)  # one-time use

# 10 — Sensitive data protection
def safe_log(message: str) -> str:
    # redact anything that looks like an API key
    message = re.sub(r'(gsk_|sk_|api[_-]?key)[a-zA-Z0-9_\-]{10,}', '[REDACTED]', message, flags=re.IGNORECASE)
    return message

# Helper for cleanup after download
def cleanup_after_interaction(output_path: Path, temp_dir: Path):
    """Deletes the output file and wipes the entire temp directory."""
    try:
        if output_path.exists():
            output_path.unlink()
            logger.info("Deleted master output: %s", output_path.name)
        
        if temp_dir.exists():
            for f in temp_dir.glob("*"):
                f.unlink(missing_ok=True)
            logger.info("Wiped temp directory: %s", temp_dir.name)
    except Exception as e:
        logger.error("Failed to clean up: %s", e)

class TTSRequest(BaseModel):
    text: str
    voice: str
    speed: float = 1.0
    format: str = "wav"
    low_ram_mode: bool = False
    humanize: bool = False
    rvc_enabled: bool = False
    rvc_model: Optional[str] = None
    rvc_index: Optional[str] = None
    rvc_pitch: int = 0
    rvc_index_rate: float = 0.75
    emotion_narration: bool = True
    smart_pronunciation: bool = True
    groq_api_key: Optional[str] = None
    temp_path: str = "./kokoro_temp/"

class ScriptRequest(BaseModel):
    source_text: str
    host1_name: str = "Eva"
    host2_name: str = "Max"

class PodcastRequest(BaseModel):
    script: str
    host1_name: str = "Eva"
    host2_name: str = "Max"
    host1_voice: str = "af_bella"
    host2_voice: str = "bm_george"
    host1_rvc: Optional[str] = None
    host1_index: Optional[str] = None
    host2_rvc: Optional[str] = None
    host2_index: Optional[str] = None
    speed: float = 1.0
    pitch: int = 0
    export_format: str = "mp3"

def get_pipeline(voice_name: str) -> EnhancedKPipeline:
    prefix = voice_name[:3].lower()
    lang_code = LANG_MAP.get(prefix, "a")
    if lang_code not in pipelines:
        logger.info("Creating pipeline for lang_code='%s'", lang_code)
        pipelines[lang_code] = EnhancedKPipeline(lang_code=lang_code, model=True)
    return pipelines[lang_code]

def validate_audio_file(path: Path) -> bool:
    try:
        p_str = str(path)
        seg = AudioSegment.from_file(p_str)
        # normalize to consistent format
        seg = seg.set_frame_rate(24000).set_channels(1).set_sample_width(2)
        seg.export(p_str, format="wav")
        return True
    except Exception as e:
        logger.warning("Invalid audio file %s: %s — skipping", path, e)
        return False

def compile_chunks(chunk_paths: List[Path], output_path: Path, crossfade_ms=8) -> Optional[Path]:
    try:
        valid_paths = [p for p in chunk_paths if validate_audio_file(p)]
        if not valid_paths:
            return None
        
        combined = AudioSegment.from_file(str(valid_paths[0]))
        for path in valid_paths[1:]:
            next_chunk = AudioSegment.from_file(str(path))
            combined = combined.append(next_chunk, crossfade=crossfade_ms)
        
        combined.export(str(output_path), format="wav")
        return output_path
    except Exception as e:
        logger.error("Compilation error: %s", e)
        return None

def split_text_into_chunks(text: str, max_words=1000) -> List[List[str]]:
    """Split text into sentences, then group into chunks of max_words."""
    # Simple sentence splitter using punctuation
    import re
    # Split paragraphs first
    paragraphs = text.split('\n\n')
    all_chunks = []
    
    for para in paragraphs:
        if not para.strip():
            continue
            
        # Split paragraph into sentences (basic regex)
        sentences = re.split(r'(?<=[.!?])\s+', para.strip())
        
        current_chunk = []
        current_word_count = 0
        
        for sentence in sentences:
            word_count = len(sentence.split())
            if current_word_count + word_count > max_words and current_chunk:
                all_chunks.append(current_chunk)
                current_chunk = [sentence]
                current_word_count = word_count
            else:
                current_chunk.append(sentence)
                current_word_count += word_count
        
        if current_chunk:
            all_chunks.append(current_chunk)
            
    return all_chunks

def convert_audio(input_path: Path, output_path: Path, export_format: str) -> Optional[Path]:
    """Converts WAV to MP3/AAC. If conversion fails, returns original WAV path."""
    try:
        if export_format.lower() == "wav":
            return input_path
        
        audio = AudioSegment.from_wav(str(input_path))
        if export_format.lower() == "mp3":
            audio.export(str(output_path), format="mp3", bitrate="192k")
        elif export_format.lower() == "aac":
            audio.export(str(output_path), format="aac", bitrate="192k")
        else:
            return input_path
            
        logger.info("Converted to %s: %s", export_format, output_path.name)
        return output_path
    except Exception as e:
        logger.warning("Conversion to %s failed: %s. Falling back to WAV.", export_format, e)
        # If conversion fails, we return the original WAV path so the user still gets audio
        return input_path

def apply_rvc(input_path: Path, output_path: Path, model_name: str, index_name: Optional[str] = None, pitch: int = 0, index_rate: float = 0.75) -> Optional[Path]:
    """Applies RVC voice conversion to an audio file."""
    try:
        model_path = RVC_MODELS_DIR / model_name
        if not model_path.exists():
            logger.warning("RVC model not found: %s", model_path)
            return input_path
            
        index_path = None
        if index_name:
            p = RVC_MODELS_DIR / index_name
            if p.exists():
                index_path = str(p)
        
        # Set parameters on the rvc instance
        rvc.f0method = "rmvpe"
        rvc.f0up_key = pitch
        rvc.index_rate = index_rate
        rvc.protect = 0.33
        
        # rvc_python expects paths in specific formats, using strings is safest
        rvc.load_model(str(model_path))
        result = rvc.infer_file(
            input_path=str(input_path),
            output_path=str(output_path)
        )
        if isinstance(result, tuple):
            audio_array, sr = result
            sf.write(str(output_path), audio_array, sr)
            
        logger.info("Applied RVC conversion using %s", model_name)
        return output_path
    except Exception as e:
        logger.warning("RVC inference failed: %s. Skipping conversion.", e)
        return input_path

def annotate_with_groq(text: str, api_key: Optional[str] = None) -> Optional[str]:
    """Uses Groq API for pacing annotation."""
    key = api_key or os.getenv("GROQ_API_KEY")
    if not key: return None
    
    try:
        client = Groq(api_key=key)
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{
                "role": "system",
                "content": """You are an audiobook narration director. Insert only timing markers into the text to create natural, human-like pacing.

Use ONLY these exact placeholders:
- [pause] — after sentences that need dramatic weight, before topic shifts
- [short_pause] — after commas in long sentences, between clauses
- [breath_pause] — at paragraph breaks, after emotionally heavy moments

Rules:
- Every paragraph break gets [breath_pause]
- Never insert more than one marker per sentence
- Return ONLY the modified text, nothing else, no explanation"""
            }, {
                "role": "user",
                "content": text
            }],
            temperature=0.3,
            max_tokens=4096
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        logger.warning("Groq annotation failed: %s", e)
        return None

def annotate_rule_based(text: str) -> str:
    sentences = re.split(r'(?<=[.!?])\s+', text)
    result = []
    sad_keywords = ["grief","loss","death","died","gone","alone","tears","cry","crying","pain","darkness","never","goodbye","farewell","empty","hollow","broken","shattered","mourning","buried","lost"]
    shock_keywords = ["suddenly","impossible","no way","couldn't believe","shocked","horrified","what","screamed","slammed","crashed","burst","exploded","froze","staggered"]
    relief_keywords = ["finally","at last","relief","safe","okay","peace","resolved","over","done","free","survived","made it","breathed","exhaled"]
    melancholy_keywords = ["silence","quiet","still","faded","distant","memory","remember","used to","once","before","never again","last time"]
    
    for sentence in sentences:
        if not sentence.strip(): continue
        lower = sentence.lower()
        if any(w in lower for w in sad_keywords) or any(w in lower for w in melancholy_keywords):
            sentence = sentence + " [breath_pause]"
        elif any(w in lower for w in shock_keywords) or sentence.strip().endswith("!"):
            sentence = sentence + " [pause]"
        elif any(w in lower for w in relief_keywords):
            sentence = sentence + " [breath_pause]"
        elif len(sentence.split()) > 20:
            sentence = sentence + " [short_pause]"
        result.append(sentence)
    
    annotated = " ".join(result)
    annotated = annotated.replace("\n\n", "\n\n[breath_pause] ")
    return annotated

def apply_smart_pronunciation(text: str) -> str:
    """Pre-processes text to fix Kokoro's common inflection and homograph errors."""
    
    # 1. Short question muting (under 4 words)
    # Prevents Kokoro from over-inflecting short questions by replacing ? with ...
    def mute_short_q(match):
        content = match.group(1)
        # Count words in the capturing group
        if len(content.strip().split()) < 4:
            return content + "..."
        return content + "?"

    # Match text from start or previous punctuation up to a question mark
    text = re.sub(r'([^.!?\n]+)\?', mute_short_q, text)

    # 2. Homograph disambiguation
    homograph_map = {
        r'\bread\b(?= the| a| an| this| that| it| him| her)': 'reed',
        r'\blead\b(?= the| a| this)': 'leed',
        r'\blead\b(?= pipe| weight| metal)': 'led',
        r'\bwound\b(?= up| around| through)': 'woond',
        r'\btear\b(?= down| apart| through)': 'tare',
        r'\btear\b(?= in| from| down her)': 'teer',
        r'\blive\b(?= in| at| here| there)': 'liv',
        r'\blive\b(?= show| music| event)': 'lyve',
    }
    
    for pattern, replacement in homograph_map.items():
        # Using a case-insensitive sub while trying to preserve leading case (naive)
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
        
    return text


def apply_pedalboard_reverb(audio: AudioSegment, room_size=0.08, wet_level=0.04) -> AudioSegment:
    """Applies reverb using Pedalboard."""
    try:
        # Convert to numpy
        samples = np.array(audio.get_array_of_samples()).astype(np.float32)
        # Normalize to -1 to 1
        max_val = np.max(np.abs(samples))
        if max_val > 0:
            samples /= max_val
            
        board = Pedalboard([Reverb(room_size=room_size, wet_level=wet_level)])
        processed = board(samples, audio.frame_rate)
        
        # Convert back
        if max_val > 0:
            processed *= max_val
            
        # Handle scaling to avoid clipping
        processed = np.clip(processed, -32768, 32767)
        int_data = processed.astype(np.int16)
        
        return AudioSegment(
            int_data.tobytes(),
            frame_rate=audio.frame_rate,
            sample_width=2,
            channels=audio.channels
        )
    except Exception as e:
        logger.warning("Reverb failed: %s", e)
        return audio


def humanize_audio(input_path: Path, output_path: Path, fmt="wav") -> Optional[Path]:
    """Applies gentle post-processing to make the audio sound more natural."""
    try:
        audio = AudioSegment.from_file(str(input_path))
        
        # 1. Slight speed nudge (1.05x sampling increase) — reduces 'read-aloud' roboticism
        audio = audio._spawn(audio.raw_data, overrides={
            "frame_rate": int(audio.frame_rate * 1.05)
        }).set_frame_rate(audio.frame_rate)
        
        # 2. Gentle low-pass warmth — roll off harsh digital highs > 13kHz
        audio = audio.low_pass_filter(13000)
        
        # 3. Apply Reverb (Requested "same room reverb as voice")
        audio = apply_pedalboard_reverb(audio)
        
        # 4. Normalize to consistent level
        audio = audio.normalize()
        
        audio.export(str(output_path), format=fmt)
        logger.info("Humanized audio: %s", output_path.name)
        return output_path
    except Exception as e:
        logger.warning("Humanize failed: %s. Returning original.", e)
        return input_path

@app.get("/voices")
async def get_voices():
    voices = list_available_voices()
    return {"voices": voices}

@app.get("/download/{filename}")
@limiter.limit("20/minute")
async def download_file(filename: str, request: Request):
    name = sanitize_filename(filename)
    path = safe_path(DEFAULT_OUTPUT_DIR, name)
    if not path.exists():
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(path)

@app.post("/generate")
@limiter.limit("10/minute;50/hour")
async def generate_tts(req: TTSRequest, request: Request):
    # This is now a wrapper for the streaming logic for backward compatibility
    # or it can remain as is but call a helper.
    # To keep it simple, I'll move the logic to a helper generator.
    async for event in generate_tts_logic(req):
        data = json.loads(event.replace("data: ", ""))
        if data.get("success"):
            return data
        if data.get("error"):
            raise HTTPException(status_code=500, detail=data["error"])
    raise HTTPException(status_code=500, detail="Unexpected end of stream")

@app.post("/generate_stream")
@limiter.limit("10/minute;50/hour")
async def generate_tts_stream(req: TTSRequest, request: Request):
    return StreamingResponse(generate_tts_logic(req), media_type="text/event-stream")

async def generate_tts_logic(req: TTSRequest):
    try:
        # Initial validation
        if not req.text.strip():
            logger.error("Text cannot be empty")
            yield f"data: {json.dumps({'error': 'Text cannot be empty'})}\n\n"
            return
        
        word_count = len(req.text.split())
        logger.info("📥 Text received — %d words", word_count)
        
        text_to_process = req.text
        
        if req.smart_pronunciation:
            text_to_process = apply_smart_pronunciation(text_to_process)
            
        req.text = sanitize_input(req.text)
        
        if req.emotion_narration:
            yield f"data: {json.dumps({'status': 'Analyzing text for pacing...'})}\n\n"
            logger.info("🧠 Sending to Groq for annotation...")
            safe_text = sanitize_for_llm(text_to_process)
            annotated = annotate_with_groq(safe_text, req.groq_api_key)
            if annotated:
                text_to_process = annotated
                marker_count = text_to_process.count('[')
                logger.info("✅ Groq returned — %d markers inserted", marker_count)
                yield f"data: {json.dumps({'mode_indicator': '🚀 Groq AI'})}\n\n"
            else:
                logger.info("⚡ Fallback: using rule-based annotation")
                text_to_process = annotate_rule_based(text_to_process)
                yield f"data: {json.dumps({'mode_indicator': '⚡ Rules'})}\n\n"
        
        # PRE-RUN CLEANUP: wipe existing temp files
        temp_dir = Path(req.temp_path).resolve()
        if temp_dir.exists():
            for f in temp_dir.glob("*"):
                try:
                    f.unlink(missing_ok=True)
                except:
                    pass
        temp_dir.mkdir(parents=True, exist_ok=True)
        
        voices = list_available_voices()
        if req.voice not in voices:
            yield f"data: {json.dumps({'error': f'Voice {req.voice} not found'})}\n\n"
            return

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_filename = f"lumina_{timestamp}"
        
        pipeline = get_pipeline(req.voice)
        voice_path = Path("voices").resolve() / f"{req.voice}.pt"

        if not req.low_ram_mode:
            # Standard Mode with Emotion/Pacing Support
            placeholders = ["[pause]", "[short_pause]", "[breath_pause]"]
            pattern = r"(\[pause\]|\[short_pause\]|\[breath_pause\])"
            parts = re.split(pattern, text_to_process)
            parts = [p for p in parts if p and p.strip()]
            logger.info("✂️ Split into %d segments", len(parts))
            
            all_audio_paths = []
            temp_dir = Path(req.temp_path).resolve()
            
            for idx, part in enumerate(parts):
                if not part or not part.strip(): continue
                
                # Check for placeholders
                if part in placeholders:
                    p_key = part.strip("[]")
                    logger.info("⏳ Injecting silence: %s", p_key)
                    temp_f = temp_dir / f"silence_{idx}.wav"
                    duration = 600 if p_key == "pause" else 300 if p_key == "short_pause" else 900
                    AudioSegment.silent(duration=duration, frame_rate=24000).export(str(temp_f), format="wav")
                    all_audio_paths.append(temp_f)
                    continue

                if "[" in part and "]" in part: continue # safeguard against missed placeholders

                logger.info("🔊 Generating segment %d/%d", idx+1, len(parts))
                yield f"data: {json.dumps({'status': f'Generating segment {len(all_audio_paths)+1}...'})}\n\n"
                wav_path = temp_dir / f"seg_{idx}.wav"
                generator = pipeline(part, voice=voice_path, speed=req.speed, split_pattern=r'\n+')
                audio_list = []
                for gs, ps, audio in generator:
                    if audio is not None:
                        if isinstance(audio, np.ndarray): audio = torch.from_numpy(audio).float()
                        audio_list.append(audio)
                
                if audio_list:
                    audio_seg = torch.cat(audio_list, dim=0)
                    sf.write(wav_path, audio_seg.numpy(), DEFAULT_SAMPLE_RATE)
                    all_audio_paths.append(wav_path)
            
            final_wav_path = DEFAULT_OUTPUT_DIR / f"{base_filename}.wav"
            logger.info("🎵 Compiling chunks into master file...")
            compile_chunks(all_audio_paths, final_wav_path)
            final_path = final_wav_path
        else:
            # Low RAM Mode with Emotion/Pacing Support
            placeholders = ["[pause]", "[short_pause]", "[breath_pause]"]
            pattern = r"(\[pause\]|\[short_pause\]|\[breath_pause\])"
            parts = re.split(pattern, text_to_process)
            parts = [p for p in parts if p and p.strip()]
            logger.info("✂️ Split into %d segments (Low RAM)", len(parts))
            
            all_final_paths = []
            block_paths = []
            temp_dir = Path(req.temp_path).resolve()
            
            warmup_sentence = None
            
            for idx, part in enumerate(parts):
                if not part or not part.strip(): continue
                
                if part in placeholders:
                    p_key = part.strip("[]")
                    logger.info("⏳ Injecting silence: %s", p_key)
                    temp_f = temp_dir / f"silence_{idx}.wav"
                    duration = 600 if p_key == "pause" else 300 if p_key == "short_pause" else 900
                    AudioSegment.silent(duration=duration, frame_rate=24000).export(str(temp_f), format="wav")
                    all_final_paths.append(temp_f)
                    continue
                
                if "[" in part and "]" in part: continue

                # Split text content into manageable chunks
                text_chunks = split_text_into_chunks(part, max_words=1000)
                for c_idx, chunk_sentences in enumerate(text_chunks):
                    logger.info("🔊 Generating segment %d/%d, chunk %d/%d", idx+1, len(parts), c_idx+1, len(text_chunks))
                    yield f"data: {json.dumps({'status': f'Generating chunk {len(all_final_paths)+1}...'})}\n\n"
                    
                    if warmup_sentence:
                        text_for_chunk = warmup_sentence + " " + " ".join(chunk_sentences)
                        is_warmup = True
                    else:
                        text_for_chunk = " ".join(chunk_sentences)
                        is_warmup = False
                    
                    warmup_sentence = chunk_sentences[-1]
                    
                    generator = pipeline(text_for_chunk, voice=voice_path, speed=req.speed, split_pattern=r'\n+')
                    segments = []
                    s_idx = 0
                    for gs, ps, audio in generator:
                        if audio is not None:
                            if is_warmup and s_idx == 0: 
                                s_idx += 1
                                continue
                            if isinstance(audio, np.ndarray): audio = torch.from_numpy(audio).float()
                            segments.append(audio)
                            s_idx += 1
                    
                    if segments:
                        chunk_f = temp_dir / f"c_{idx}_{c_idx}.wav"
                        sf.write(chunk_f, torch.cat(segments, dim=0).numpy(), DEFAULT_SAMPLE_RATE)
                        all_final_paths.append(chunk_f)
                        del segments
                        import gc
                        gc.collect()
                        if torch.cuda.is_available(): torch.cuda.empty_cache()

                # Rolling compilation for very long files
                if len(all_final_paths) >= 20:
                    block_num = len(block_paths)
                    logger.info("🎵 Compiling block %d", block_num)
                    b_file = temp_dir / f"b_{block_num}.wav"
                    compile_chunks(all_final_paths, b_file)
                    block_paths.append(b_file)
                    for p in all_final_paths:
                        # Only delete temp chunk files, not compiled blocks
                        if p != b_file and p.parent == temp_dir: 
                            p.unlink(missing_ok=True)
                    all_final_paths = []

            if all_final_paths:
                b_file = temp_dir / f"b_{len(block_paths)}.wav"
                compile_chunks(all_final_paths, b_file)
                block_paths.append(b_file)

            final_wav_path = DEFAULT_OUTPUT_DIR / f"{base_filename}.wav"
            if len(block_paths) == 1:
                shutil.move(str(block_paths[0]), str(final_wav_path))
            else:
                compile_chunks(block_paths, final_wav_path)
            
            # Cleanup blocks
            for p in block_paths:
                if p.exists(): p.unlink()
            
            final_path = final_wav_path

        if not final_path.exists() or final_path.stat().st_size == 0:
            logger.error("❌ Compilation failed — master file missing, aborting pipeline")
            yield f"data: {json.dumps({'error': 'Compilation failed'})}\n\n"
            raise HTTPException(status_code=500, detail="Compilation failed")

        # Acoustic Humanizer
        if req.humanize:
            logger.info("🎭 Running acoustic humanizer...")
            yield f"data: {json.dumps({'status': 'Applying acoustic humanizer...'})}\n\n"
            h_path = DEFAULT_OUTPUT_DIR / f"h_{final_path.name}"
            res = humanize_audio(final_path, h_path)
            if res: final_path = res

        # RVC Post-Processing
        if req.rvc_enabled and req.rvc_model:
            logger.info("🎙️ Running RVC conversion...")
            yield f"data: {json.dumps({'status': 'Applying RVC voice conversion...'})}\n\n"
            rvc_out = DEFAULT_OUTPUT_DIR / f"rvc_{final_path.name}"
            # Ensure it's wav for RVC
            res = apply_rvc(final_path, rvc_out, req.rvc_model, req.rvc_index, req.rvc_pitch, req.rvc_index_rate)
            if res: final_path = res

        # Convert
        if req.format.lower() != "wav":
            yield f"data: {json.dumps({'status': f'Converting to {req.format}...'})}\n\n"
            target_path = DEFAULT_OUTPUT_DIR / f"{base_filename}.{req.format.lower()}"
            converted = convert_audio(final_path, target_path, req.format.lower())
            if converted:
                final_path = converted

        # Final notification with download link
        logger.info("✅ Final export complete — %s", final_path.name)
        event_data = {
            'success': True, 
            'url': f'/outputs/{final_path.name}', 
            'download_url': f'/download/{final_path.name}',
            'filename': final_path.name
        }
        yield f"data: {json.dumps(event_data)}\n\n"

    except Exception as e:
        logger.error("❌ ERROR: %s", str(e))
        yield f"data: {json.dumps({'error': str(e)})}\n\n"

@app.get("/download/{filename}")
async def download_file(filename: str, background_tasks: BackgroundTasks):
    path = DEFAULT_OUTPUT_DIR / filename
    if not path.exists():
        raise HTTPException(status_code=404, detail="File not found")
    
    # Schedule self-destruction after response is sent
    background_tasks.add_task(cleanup_after_interaction, path, TEMP_DIR)
    
    return FileResponse(
        path, 
        media_type='application/octet-stream', 
        filename=filename
    )

@app.post("/upload_model")
@limiter.limit("5/hour")
async def upload_model(request: Request, file: UploadFile = File(...)):
    name, contents = await validate_upload(file, {'.pth', '.index'}, max_mb=100)
    target = safe_path(RVC_MODELS_DIR, name)
    with target.open("wb") as buffer:
        buffer.write(contents)
    return {"filename": name, "status": "success"}

@app.get("/rvc_models")
async def list_rvc_models():
    models = [f.name for f in RVC_MODELS_DIR.glob("*.pth")]
    indices = [f.name for f in RVC_MODELS_DIR.glob("*.index")]
    return {"models": models, "indices": indices}

@app.get("/llm_status")
async def llm_status():
    # Force reload .env to catch manual edits without server restart
    load_dotenv(override=True)
    key = os.getenv("GROQ_API_KEY")
    backend = "groq" if key else "none"
    logger.info("LLM Status check: detected key=%s -> backend=%s", 
                "EXISTS" if key else "MISSING", backend)
    return {"backend": backend}

@app.post("/save_groq_key")
async def save_groq_key(request: Request):
    data = await request.json()
    key = data.get("key")
    if key:
        os.environ["GROQ_API_KEY"] = key
        # Also try to update .env if it exists
        if os.path.exists(".env"):
            with open(".env", "r") as f:
                lines = f.readlines()
            new_lines = []
            found = False
            for line in lines:
                if line.startswith("GROQ_API_KEY="):
                    new_lines.append(f"GROQ_API_KEY={key}\n")
                    found = True
                else:
                    new_lines.append(line)
            if not found:
                new_lines.append(f"\nGROQ_API_KEY={key}\n")
            with open(".env", "w") as f:
                f.writelines(new_lines)
        return {"success": True}
    return {"success": False}

async def generate_script_with_best_backend(source_text: str, host1_name: str, host2_name: str):
    key = os.getenv("GROQ_API_KEY")
    if not key:
        return None, None
        
    prompt = f"""You are a podcast script writer. Convert the source material into a natural, engaging two-host podcast conversation between {host1_name} and {host2_name}.

Format EVERY line EXACTLY like this — no exceptions:
[{host1_name.lower()}]: dialogue here
[{host2_name.lower()}]: dialogue here

Personality:
- {host1_name} is curious, asks questions, reacts with wonder
- {host2_name} is knowledgeable, explains clearly, uses good analogies
- Include natural reactions: "right", "exactly", "oh interesting", "wait so..."
- No monologues longer than 3 sentences — keep it back and forth
- Start with both hosts introducing the topic
- End with a closing thought from each host

CRITICAL: Return ONLY the formatted script. No preamble, no explanation, nothing else.

Source material:
{source_text}"""

    logger.info("🚀 Using Groq for script generation")
    try:
        from groq import Groq
        client = Groq(api_key=key)
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=4096
        )
        script = response.choices[0].message.content.strip()
        return script, "groq"
    except Exception as e:
        logger.error("Groq failed: %s", e)
        return None, None

@app.post("/generate_script")
@limiter.limit("5/minute;20/hour")
async def generate_script(body: ScriptRequest, request: Request):
    logger.info("📥 [API] /generate_script hit")
    # temporarily skip CSRF for debugging
    # validate_csrf(request) 
    
    try:
        logger.info("  -> Sanitizing input...")
        source_text = sanitize_input(body.source_text)
        source_text = sanitize_for_llm(source_text)
        
        logger.info("  -> Data: Host1=%s, Host2=%s, text length=%d", 
            body.host1_name, body.host2_name, len(source_text))
            
        logger.info("  -> Requesting script from LLM...")
        script, backend = await generate_script_with_best_backend(
            source_text,
            body.host1_name, 
            body.host2_name
        )
        
        if script is None:
            logger.error("  -> LLM returned None (no backend or failure)")
            raise HTTPException(status_code=503, detail="no_llm_available")
            
        logger.info("  -> Script generated successfully via %s", backend)
        return {"script": script, "backend": backend}
    except HTTPException:
        raise
    except Exception as e:
        logger.error("  -> EXCEPTION in generate_script: %s", str(e), exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/generate_podcast")
@limiter.limit("3/minute;10/hour")
async def generate_podcast_stream(req: PodcastRequest, request: Request):
    return StreamingResponse(generate_podcast_logic(req), media_type="text/event-stream")

async def generate_podcast_logic(req: PodcastRequest):
    """Generate a full podcast from a formatted script with two distinct voices."""
    import time
    try:
        lines = [l.strip() for l in req.script.split('\n') if l.strip()]
        if not lines:
            yield f"data: {json.dumps({'error': 'Script is empty'})}\n\n"
            return
        
        total = len(lines)
        logger.info("🎙️ Podcast generation started — %d lines", total)
        yield f"data: {json.dumps({'status': f'Starting podcast — {total} lines'})}\n\n"
        
        temp_dir = TEMP_DIR / f"podcast_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        temp_dir.mkdir(parents=True, exist_ok=True)
        
        all_clip_paths = []
        
        for i, line in enumerate(lines):
            line_lower = line.lower().strip()
            # Determine speaker (case-insensitive normalization)
            if line_lower.startswith(f'[{req.host1_name.lower()}]:'):
                text = line[line.index(':')+1:].strip()
                voice = req.host1_voice
                rvc_model = req.host1_rvc
                rvc_index = req.host1_index
                speaker = req.host1_name
            elif line_lower.startswith(f'[{req.host2_name.lower()}]:'):
                text = line[line.index(':')+1:].strip()
                voice = req.host2_voice
                rvc_model = req.host2_rvc
                rvc_index = req.host2_index
                speaker = req.host2_name
            else:
                logger.warning("Line %d: No valid speaker found, skipping: %s", i+1, line[:20])
                continue
            
            if not text.strip():
                continue
                
            logger.info("🎙️ %s — line %d/%d", speaker, i+1, total)
            yield f"data: {json.dumps({'status': f'🎙️ {speaker} — line {i+1}/{total}'})}\n\n"
            
            # Generate TTS
            pipeline = get_pipeline(voice)
            voice_path = Path("voices").resolve() / f"{voice}.pt"
            generator = pipeline(text, voice=voice_path, speed=req.speed, split_pattern=r'\n+')
            
            audio_list = []
            for gs, ps, audio in generator:
                if audio is not None:
                    if isinstance(audio, np.ndarray):
                        audio = torch.from_numpy(audio).float()
                    audio_list.append(audio)
            
            if not audio_list:
                continue
            
            line_wav = temp_dir / f"line_{i:04d}.wav"
            audio_data = torch.cat(audio_list, dim=0)
            sf.write(str(line_wav), audio_data.numpy(), DEFAULT_SAMPLE_RATE)
            
            # Apply RVC if model provided
            if rvc_model and rvc_model not in ("none", ""):
                rvc_out = temp_dir / f"rvc_{i:04d}.wav"
                res = apply_rvc(line_wav, rvc_out, rvc_model, rvc_index, req.pitch)
                if res and res != line_wav:
                    line_wav = res
            
            all_clip_paths.append(line_wav)
            
            # Insert silence gap between lines
            gap_wav = temp_dir / f"gap_{i:04d}.wav"
            if i < len(lines) - 1:
                next_line = lines[i+1] if i+1 < len(lines) else ""
                current_is_host1 = line.startswith(f'[{req.host1_name}]')
                next_is_host1 = next_line.startswith(f'[{req.host1_name}]')
                if current_is_host1 != next_is_host1:
                    # Speaker switch — longer gap
                    AudioSegment.silent(duration=400, frame_rate=24000).export(str(gap_wav), format="wav")
                else:
                    # Same speaker — shorter gap
                    AudioSegment.silent(duration=150, frame_rate=24000).export(str(gap_wav), format="wav")
                all_clip_paths.append(gap_wav)
            
            # Thermal gap
            time.sleep(0.8)
        
        if not all_clip_paths:
            yield f"data: {json.dumps({'error': 'No audio was generated'})}\n\n"
            return
        
        # Compile all clips
        logger.info("🎵 Compiling podcast master...")
        yield f"data: {json.dumps({'status': 'Compiling podcast...'})}\n\n"
        
        master_wav = DEFAULT_OUTPUT_DIR / f"podcast_{datetime.now().strftime('%Y%m%d_%H%M%S')}.wav"
        compile_chunks(all_clip_paths, master_wav, crossfade_ms=5)
        
        if not master_wav.exists() or master_wav.stat().st_size == 0:
            yield f"data: {json.dumps({'error': 'Compilation failed'})}\n\n"
            return
        
        # Humanize
        logger.info("🎭 Humanizing podcast audio...")
        yield f"data: {json.dumps({'status': 'Applying humanizer...'})}\n\n"
        h_path = DEFAULT_OUTPUT_DIR / f"h_{master_wav.name}"
        res = humanize_audio(master_wav, h_path)
        final_path = res if res else master_wav
        
        # Convert format
        if req.export_format.lower() != "wav":
            yield f"data: {json.dumps({'status': f'Converting to {req.export_format}...'})}\n\n"
            target = DEFAULT_OUTPUT_DIR / f"podcast_{datetime.now().strftime('%Y%m%d_%H%M%S')}.{req.export_format.lower()}"
            converted = convert_audio(final_path, target, req.export_format.lower())
            if converted:
                final_path = converted
        
        # Cleanup temp
        for p in temp_dir.glob("*"):
            p.unlink(missing_ok=True)
        temp_dir.rmdir()
        
        # Cleanup intermediate wav files 
        if master_wav.exists() and final_path != master_wav:
            master_wav.unlink(missing_ok=True)
        if h_path.exists() and final_path != h_path:
            h_path.unlink(missing_ok=True)
        
        logger.info("✅ Podcast export complete — %s", final_path.name)
        yield f"data: {json.dumps({'success': True, 'filename': final_path.name})}\n\n"
    
    except Exception as e:
        logger.error("❌ Podcast error: %s", str(e))
        yield f"data: {json.dumps({'error': str(e)})}\n\n"


@app.get("/", response_class=HTMLResponse)
async def serve_index():
    with open("index.html", "r", encoding="utf-8") as f:
        return f.read()

if __name__ == "__main__":

    # Pre-build model for default English
    logger.info("Pre-building default English pipeline...")
    get_pipeline("af_bella")
    uvicorn.run(app, host="0.0.0.0", port=8000)
