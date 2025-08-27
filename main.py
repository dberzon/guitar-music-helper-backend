import logging

# Module-level logger for consistent logging across the module
logger = logging.getLogger(__name__)

from typing import Optional


def _normalize_madmom_label(label: str) -> Optional[str]:
    lbl = str(label)
    if lbl in ("N", "X", "no_chord"):
        return None
    if ":" not in lbl:
        return lbl  # already simple
    root, qual = lbl.split(":", 1)
    q = qual.lower()
    if q.startswith("maj"):
        return root
    if q.startswith("min"):
        return f"{root}m"
    if q.startswith("dim"):
        return f"{root}dim"
    if q.startswith("aug"):
        return f"{root}aug"
    if q.startswith("sus2"):
        return f"{root}sus2"
    if q.startswith("sus4"):
        return f"{root}sus4"
    if q.startswith("maj7"):
        return f"{root}maj7"
    if q.startswith("min7"):
        return f"{root}m7"
    if q.startswith("7"):
        return f"{root}7"
    return f"{root}:{qual}"
# --- Madmom Chord Recognition Integration ---
def detect_chords_madmom(audio_path: str):
    """Use Madmom's state-of-the-art chord detection with label normalization."""
    # Prefer pre-instantiated processors when available for performance.
    global MADMOM_CNN_PROCESSOR, MADMOM_CRF_PROCESSOR
    if MADMOM_CNN_PROCESSOR is not None and MADMOM_CRF_PROCESSOR is not None:
        cnn_proc = MADMOM_CNN_PROCESSOR
        crf_proc = MADMOM_CRF_PROCESSOR
    else:
        # Try a lazy import + instantiation for environments where module-level
        # pre-instantiation wasn't possible (e.g., missing weights or transient
        # environment issues).
        try:
            from madmom.features.chords import CNNChordFeatureProcessor, CRFChordRecognitionProcessor
            cnn_proc = CNNChordFeatureProcessor()
            crf_proc = CRFChordRecognitionProcessor()
        except (ImportError, ModuleNotFoundError) as e:
            logger.debug(f"Madmom not available: {e}")
            return []

    try:
        feats = cnn_proc(audio_path)
        segs = crf_proc(feats)
        out = []
        for start, end, label in segs:
            name = _normalize_madmom_label(label)
            if not name:
                continue
            out.append({"time": float(start), "duration": float(end - start), "chord": name})
        return out
    except (RuntimeError, ValueError, OSError) as e:
        logger.warning(f"Madmom chord detection failed during processing: {e}")
        return []

import os
import tempfile
import base64
import binascii
import time
import asyncio
import uuid
import subprocess
# sys, platform, and gc are imported locally where needed to avoid global overhead
import re
import shutil
from typing import Dict, TypedDict, List, Optional, Union, Literal
from pathlib import Path
from contextlib import asynccontextmanager
import hashlib
from contextvars import ContextVar

# --- Import helpers for file handling and hashing ---
try:
    from .file_utils import temporary_audio_file, get_file_hash
except ImportError:
    # fallback for monolithic file or if not present
    import aiofiles
@asynccontextmanager
async def temporary_audio_file(file):
    suffix = Path(file.filename).suffix
    fd, tmp_path = tempfile.mkstemp(suffix=suffix)
    os.close(fd)
    try:
        await file.seek(0)
        async with aiofiles.open(tmp_path, 'wb') as out:
            while chunk := await file.read(1024 * 1024):
                await out.write(chunk)
        yield tmp_path
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass

# Utility: file hashing available at module scope
def get_file_hash(path):
    """Return SHA256 hash of a file."""
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            h.update(chunk)
    return h.hexdigest()

# --- Handle optional madmom import gracefully ---
try:
    import madmom
except ImportError:
    madmom = None

# Module-level Madmom processors (singletons) - try to instantiate once for throughput.
MADMOM_CNN_PROCESSOR = None
MADMOM_CRF_PROCESSOR = None
try:
    if madmom is not None:
        from madmom.features.chords import CNNChordFeatureProcessor, CRFChordRecognitionProcessor
        try:
            MADMOM_CNN_PROCESSOR = CNNChordFeatureProcessor()
            MADMOM_CRF_PROCESSOR = CRFChordRecognitionProcessor()
            logger.info("Madmom processors instantiated at module import")
        except Exception:
            MADMOM_CNN_PROCESSOR = None
            MADMOM_CRF_PROCESSOR = None
except Exception:
    MADMOM_CNN_PROCESSOR = None
    MADMOM_CRF_PROCESSOR = None

from fastapi import FastAPI, File, UploadFile, HTTPException, Request, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import importlib.util as _importlib_util
from shutil import which as _which

# --- New: Background jobs + YouTube support ---
from redis import Redis, ConnectionPool
from rq import Queue, Retry
from pydantic import BaseModel, Field, HttpUrl, field_validator, model_validator
import yt_dlp
from base64 import b64decode
import uvicorn
import requests
from pydantic_settings import BaseSettings
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.middleware import SlowAPIMiddleware
from slowapi.errors import RateLimitExceeded

# Memory monitoring
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    psutil = None

# Runtime-safe import of the processing helper, with explicit logging
try:
    from transcription_utils import process_basic_pitch_output  # real implementation
    logger.info("Using real process_basic_pitch_output from transcription_utils.")
except (ImportError, ModuleNotFoundError) as e:
    logger.warning(f"Using fallback process_basic_pitch_output (no-op) — accuracy will be poor. Import error: {e}")
    def process_basic_pitch_output(model_output, midi_data, note_events, sr, duration):
        return {"chords": [], "melody": [], "tempo": None}

# --- Types ---
class ProcessingResult(TypedDict):
    """Type definition for audio processing results"""
    metadata: Dict[str, Union[float, int]]
    chords: List[Dict[str, Union[str, float]]]
    melody: List[Dict[str, Union[str, float]]]
    tempo: Optional[float]  # Tempo is typically a single float (BPM)


# --- Pydantic response models (used by FastAPI to document responses) ---
class ChordSegment(BaseModel):
    time: float
    duration: float
    chord: str


class MetadataModel(BaseModel):
    filename: Optional[str] = None
    processingTime: Optional[float] = None
    duration: Optional[float] = None
    sampleRate: Optional[int] = None
    source: Optional[str] = None
    videoId: Optional[str] = None
    title: Optional[str] = None
    uploader: Optional[str] = None
    webpage_url: Optional[str] = None


class TranscribeResponse(BaseModel):
    metadata: MetadataModel
    chords: List[ChordSegment]
    melody: List[dict]
    tempo: Optional[dict] = None
    debug: Optional[dict] = None

# --- Constants ---
ALLOWED_MIME_TYPES = {
    '.wav': ['audio/wav', 'audio/x-wav'],
    '.mp3': ['audio/mpeg', 'audio/mp3', 'application/octet-stream'],  # Allow octet-stream for curl uploads
    '.m4a': ['audio/x-m4a', 'audio/m4a', 'audio/mp4', 'application/octet-stream'],
    '.flac': ['audio/flac', 'audio/x-flac', 'application/octet-stream'],
    '.ogg': ['audio/ogg', 'application/ogg', 'application/octet-stream']
}

# Audio preprocessing defaults (pin sample rate to 22050 for best accuracy)
TARGET_SR = 22050  # Pin to 22050 for Basic Pitch accuracy
MAX_AUDIO_DURATION_S = int(os.getenv("MAX_AUDIO_SECONDS", "900"))  # 15 minutes

# Validate/sanitize caps
if MAX_AUDIO_DURATION_S <= 0:
    MAX_AUDIO_DURATION_S = 900

# --- Memory Management Utilities ---
def get_memory_usage() -> float:
    """Get current memory usage in MB"""
    if not PSUTIL_AVAILABLE:
        return 0.0
    try:
        process = psutil.Process()
        return process.memory_info().rss / 1024 / 1024
    except Exception:
        return 0.0

def check_memory_availability(estimated_mb: float) -> bool:
    """Check if there's enough memory available"""
    if not PSUTIL_AVAILABLE:
        return True  # Assume OK if psutil not available
    try:
        current_usage = get_memory_usage()
        available = psutil.virtual_memory().available / 1024 / 1024
        return available > estimated_mb * 1.5  # 50% buffer
    except Exception:
        return True  # Assume OK on error

# --- Simple Metrics Collection ---
from datetime import datetime
from collections import defaultdict

class SimpleMetrics:
    """Simple in-memory metrics collection for monitoring"""
    def __init__(self):
        self.request_count = defaultdict(int)
        self.error_count = defaultdict(int) 
        self.processing_times = []
        
    def record_request(self, endpoint: str):
        """Record a request to an endpoint"""
        self.request_count[endpoint] += 1
        
    def record_error(self, endpoint: str, error_type: str):
        """Record an error for an endpoint"""
        self.error_count[f"{endpoint}:{error_type}"] += 1
        
    def record_processing_time(self, endpoint: str, duration: float):
        """Record processing time for an endpoint"""
        self.processing_times.append((datetime.now(), endpoint, duration))
        # Keep only last 100 entries to prevent memory growth
        if len(self.processing_times) > 100:
            self.processing_times = self.processing_times[-100:]
    
    def get_stats(self) -> dict:
        """Get current metrics statistics"""
        avg_time = 0.0
        if self.processing_times:
            avg_time = sum(t[2] for t in self.processing_times) / len(self.processing_times)
        
        return {
            "requests": dict(self.request_count),
            "errors": dict(self.error_count),
            "avg_processing_time_seconds": round(avg_time, 2),
            "total_requests": sum(self.request_count.values()),
            "total_errors": sum(self.error_count.values()),
            "current_memory_usage_mb": get_memory_usage() if PSUTIL_AVAILABLE else None
        }


# --- Audio Processing Orchestrator ---


# --- Configure Logging ---
# Initialize config first to get LOG_LEVEL
from pydantic_settings import SettingsConfigDict

class Config(BaseSettings):
    MAX_FILE_SIZE_MB: int = Field(50, description="Max file size in MB - configurable via environment")
    ALLOWED_EXTENSIONS: set[str] = {'.wav', '.mp3', '.m4a', '.flac', '.ogg'}
    MAX_WORKERS: int = 1  # Reduced to prevent memory exhaustion
    ENVIRONMENT: str = Field("development", env="RAILWAY_ENVIRONMENT")
    LOG_LEVEL: str = "INFO"
    PROCESSING_TIMEOUT: int = Field(45, description="Max processing time in seconds - increased for Railway")
    # Cache
    CACHE_TTL_SECONDS: int = Field(3600, description="Seconds to cache identical-file results")
    CACHE_MAX_ITEMS: int = Field(50, description="Max cached entries")
    # YouTube safety guard
    MAX_YOUTUBE_DURATION_SEC: int = Field(15 * 60, description="Reject YouTube videos longer than this (seconds)")
    
    # Dynamic CORS configuration based on environment
    CORS_ORIGINS: list[str] = Field(
        default_factory=lambda: [
            "http://localhost:5173", 
            "http://127.0.0.1:5173", 
            "http://localhost:5174", 
            "http://127.0.0.1:5174",
            "http://localhost:3000",
            "http://127.0.0.1:3000",
            # Vercel stable production domain
            "https://guitar-music-helper.vercel.app",
            # Current Vercel preview domain
            "https://guitar-music-helper-7g9fwzvch-dberzons-projects.vercel.app",
            # Previous example domains (keep for compatibility)
            "https://guitar-music-helper-h7erfkcq4-dberzons-projects.vercel.app",
            "https://guitar-music-helper-h9i35sgkq-dberzons-projects.vercel.app"
        ],
        description="List of allowed CORS origins"
    )
    CORS_ORIGIN_REGEX: str | None = Field(
        default=r"^https://.*-dberzon-projects\.vercel\.app$",
        description="Regex for Vercel preview deployments"
    )
    
    # Backend identification for environment detection
    BACKEND_URL: str = Field(
        default="http://localhost:8000",
        description="Backend URL for this instance"
    )
    
    # Development mode settings
    ENABLE_DEBUG_ENDPOINTS: bool = Field(
        default=False,
        description="Enable debug endpoints in development"
    )

    # Modern Pydantic v2 configuration - ignore extra environment variables
    model_config = SettingsConfigDict(
        env_file=".env",
        extra='ignore'  # Ignore extra environment variables to prevent errors
    )

    @property
    def MAX_FILE_SIZE(self):
        return self.MAX_FILE_SIZE_MB * 1024 * 1024
    
    @property
    def is_development(self) -> bool:
        """Check if running in development mode"""
        return self.ENVIRONMENT.lower() in ["development", "dev", "local"]
    
    @property
    def is_production(self) -> bool:
        """Check if running in production mode"""
        return self.ENVIRONMENT.lower() in ["production", "prod"]

    @property
    def cache_enabled(self) -> bool:
        """Enable caching only in production by default"""
        return self.is_production

    @field_validator('MAX_FILE_SIZE_MB')
    def validate_max_file_size(cls, v):
        if v < 1 or v > 100:
            raise ValueError('MAX_FILE_SIZE_MB must be between 1 and 100')
        return v
    
    @field_validator('PROCESSING_TIMEOUT')
    def validate_timeout(cls, v):
        if v < 10 or v > 300:
            raise ValueError('PROCESSING_TIMEOUT must be between 10 and 300 seconds')
        return v

config = Config()

# Update the CORS_ORIGINS to include the Railway production URL
config.CORS_ORIGINS.append("https://web-production-84b20.up.railway.app")

# --- Error types and dependency flags (minimal definitions) ---
class AudioProcessingError(Exception):
    pass

class DependencyError(Exception):
    pass

class ProcessingTimeoutError(AudioProcessingError):
    pass

# Runtime flags indicating optional ML dependencies and models
DEPENDENCIES_LOADED = False
MODELS_LOADED = False

# Attempt to detect optional ML dependencies without importing heavy modules
from importlib.util import find_spec as _find_spec
DEPENDENCIES_LOADED = all(_find_spec(n) is not None for n in ("librosa", "numpy", "basic_pitch"))
MODELS_LOADED = DEPENDENCIES_LOADED


def require_debug_enabled():
    """Raise 404 if debug endpoints are disabled in configuration."""
    if not config.ENABLE_DEBUG_ENDPOINTS:
        raise HTTPException(status_code=404, detail="Not found")

# Initialize rate limiter - limits requests based on client IP address
limiter = Limiter(key_func=get_remote_address)

def job_rate_key(request: Request):
    """Custom rate limiting key that combines IP address with job ID for per-job rate limiting"""
    return f"{get_remote_address(request)}:{request.path_params.get('job_id','')}"
 

# Configure logging using the config LOG_LEVEL
request_context: ContextVar[dict] = ContextVar("request_context", default={})

logging.basicConfig(
    level=config.LOG_LEVEL.upper(),
    format="%(asctime)s %(levelname)s [%(request_id)s] %(name)s: %(message)s"
)

class ContextFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        ctx = request_context.get({})
        record.request_id = ctx.get("request_id", "no-req")
        return True

for handler in logging.getLogger().handlers:
    handler.addFilter(ContextFilter())
logger = logging.getLogger(__name__)

YOUTUBE_REGEX = r'^(https?://)?(www\.)?(youtube\.com|youtu\.be)/'

def is_youtube_url(u: str) -> bool:
    return re.match(YOUTUBE_REGEX, u or "", flags=re.IGNORECASE) is not None

def _prepare_cookiefile_from_env(tmp_dir: str) -> Optional[str]:
    """
    Optionally prepare a cookies.txt file from env vars:
      - YT_COOKIES: raw Netscape-format cookies.txt text
      - YT_COOKIES_B64: base64-encoded cookies.txt
    Returns path or None.
    """
    raw = os.getenv("YT_COOKIES")
    b64 = os.getenv("YT_COOKIES_B64")
    if not raw and not b64:
        return None
    path = os.path.join(tmp_dir, "cookies.txt")
    try:
        data = raw if raw else b64decode(b64).decode("utf-8", errors="ignore")
        with open(path, "w", encoding="utf-8") as f:
            f.write(data)
        return path
    except (binascii.Error, ValueError, UnicodeDecodeError, OSError) as e:
        logging.warning("Failed to load cookies from env: %s", e)
        return None

def yt_download_to_wav(tmp_dir: str, url: str) -> tuple[str, dict]:
    """Download YT audio with yt-dlp → mono 22.05 kHz WAV. Hardened for cloud IPs."""
    out_tmpl = os.path.join(tmp_dir, "%(id)s.%(ext)s")
    # Optional cookie support (exported from your browser, base64 in env)
    cookiefile_path = _prepare_cookiefile_from_env(tmp_dir)
    proxy = os.getenv("YTDLP_PROXY")  # e.g. http://user:pass@host:port  (optional)

    base_opts = {
        "format": "bestaudio/best",
        "outtmpl": out_tmpl,
        "noplaylist": True,
        "quiet": True,
        "retries": 3,
        "fragment_retries": 3,
        "concurrent_fragment_downloads": 1,
        "socket_timeout": 30,
        "forceipv4": True,            # IPv6 sometimes blocked on PaaS
        "geo_bypass": True,
        "http_headers": {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
            "Accept-Language": "en-US,en;q=0.9",
            "Referer": "https://www.youtube.com/",
        },
        "postprocessors": [
            {"key": "FFmpegExtractAudio", "preferredcodec": "wav", "preferredquality": "0"}
        ],
        "postprocessor_args": ["-ar", str(TARGET_SR), "-ac", "1"],
        "prefer_ffmpeg": True,
    }
    if proxy:
        base_opts["proxy"] = proxy
    if cookiefile_path:
        base_opts["cookiefile"] = cookiefile_path

    # Probe WITHOUT download to enforce your guards
    with yt_dlp.YoutubeDL(dict(base_opts)) as ydl:
        info = ydl.extract_info(url, download=False)
        dur = int(info.get("duration") or 0)
        if info.get("is_live"):
            raise ValueError("Livestreams are not supported.")
        if dur and dur > config.MAX_YOUTUBE_DURATION_SEC:
            raise ValueError(f"Video too long: {dur}s exceeds limit of {config.MAX_YOUTUBE_DURATION_SEC}s")

    # Try a couple of player clients to dodge some bot checks
    clients = os.getenv("YTDLP_PLAYER_CLIENTS", "android,web_safari").split(",")
    last_err = None
    for client in [c.strip() for c in clients if c.strip()]:
        try:
            ydl_opts = dict(base_opts)
            ydl_opts["extractor_args"] = {"youtube": {"player_client": [client]}}
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=True)
                vid = info.get("id")
                wav_path = os.path.join(tmp_dir, f"{vid}.wav")
                if not os.path.exists(wav_path):
                    in_file = ydl.prepare_filename(info)
                    subprocess.run(["ffmpeg", "-y", "-i", in_file, "-ar", str(TARGET_SR), "-ac", "1", wav_path], check=True)
                return wav_path, info
        except (yt_dlp.utils.DownloadError, subprocess.CalledProcessError, RuntimeError) as e:
            last_err = e
            # If bot/CAPTCHA message seen and we had no cookies, hint that cookies are required
            err_str = str(e)
            if ("not a bot" in err_str.lower() or "confirm you're not a bot" in err_str.lower()) and not cookiefile_path:
                raise RuntimeError("YouTube challenged this IP. Add cookies via YTDLP_COOKIES_B64.") from e
            continue
    raise last_err if last_err else RuntimeError("yt-dlp failed without error")

def process_youtube_job(url: str) -> dict:
    tmp_dir = tempfile.mkdtemp(prefix="yt_")
    try:
        audio_path, info = yt_download_to_wav(tmp_dir, url)
        result = process_audio_file_sync(audio_path)
        tempo = result.get("tempo")
        return {
            "metadata": {
                "filename": f"{info.get('id')}.wav",
                "processingTime": result.get("metadata", {}).get("processingTime"),
                **result.get("metadata", {}),
                "source": "youtube",
                "videoId": info.get("id"),
                "title": info.get("title"),
                "uploader": info.get("uploader"),
                "webpage_url": info.get("webpage_url"),
            },
            "chords": result.get("chords", []),
            "melody": result.get("melody", []),
            "tempo": tempo if isinstance(tempo, dict) else ({"bpm": tempo} if tempo else None),
        }
    except yt_dlp.utils.DownloadError as e:
        logger.error(f"yt-dlp download error for URL={url}: {e}", exc_info=True)
        # Surface a meaningful, stable message to the UI
        raise AudioProcessingError(f"YouTube download failed: {e}") from e
    except subprocess.CalledProcessError as e:
        logger.error(f"ffmpeg conversion failed for URL={url}: {e}", exc_info=True)
        raise AudioProcessingError("ffmpeg failed while converting audio to WAV") from e
    except Exception as e:
        logger.error(f"Unexpected error in process_youtube_job for URL={url}: {e}", exc_info=True)
        msg = str(e)
        # Give the frontend a friendly, actionable error
        if "Sign in to confirm you're not a bot" in msg or "confirm you're not a bot" in msg:
            raise AudioProcessingError(
                "YouTube blocked the request (bot/age verification). "
                "Try another URL, or add cookies: set env YT_COOKIES (raw Netscape cookies.txt) "
                "or YT_COOKIES_B64 (base64 of cookies.txt) in Railway, then redeploy."
            ) from e
        raise AudioProcessingError(f"YouTube download failed: {msg}") from e
    finally:
        try:
            import shutil
            shutil.rmtree(tmp_dir, ignore_errors=True)
        except OSError:
            pass


class YouTubeJob(BaseModel):
    url: str

# -------------------------------
# Request validation models
# -------------------------------
class TranscribeRequest(BaseModel):
    kind: Literal["file","url","youtube"]
    url: Optional[HttpUrl] = None
    youtube: Optional[str] = Field(None, max_length=255)
    immediate: Optional[bool] = False

    class Config:
        max_anystr_length = 255

    @model_validator(mode="after")
    def _validate_one_of(self):
        if self.kind == "url" and not self.url:
            raise ValueError("url is required when kind=url")
        if self.kind == "youtube" and not self.youtube:
            raise ValueError("youtube id/url is required when kind=youtube")
        return self

# -------------------------------
# Direct URL transcription (BG job)
# -------------------------------
class DirectURLJob(BaseModel):
    url: str

_HTTP_URL_RE = re.compile(r"^https?://", re.IGNORECASE)
MAX_DOWNLOAD_BYTES = 50 * 1024 * 1024  # 50 MB hard cap for direct URL ingestion

def _is_http_url(u: Optional[str]) -> bool:
    return bool(u and _HTTP_URL_RE.match(u))

def _sanitize_filename(name: str) -> str:
    # Keep it simple: strip path bits and replace suspicious chars
    base = os.path.basename(name or "")
    return re.sub(r"[^A-Za-z0-9._-]+", "_", base) or "remote_audio"

def _download_to_wav(tmp_dir: str, url: str) -> str:
    """
    Stream-download the remote file with a size cap, then normalize to 22.05kHz mono WAV.
    Returns absolute path to the WAV file.
    """
    # HEAD request to check content type and size before downloading
    try:
        head_resp = requests.head(url, timeout=10, allow_redirects=True)
        head_resp.raise_for_status()

        # Check Content-Length if available
        content_length = head_resp.headers.get("content-length")
        if content_length and int(content_length) > MAX_DOWNLOAD_BYTES:
            raise ValueError(f"File too large: {content_length} bytes exceeds {MAX_DOWNLOAD_BYTES} byte limit")

        # Check Content-Type if available (allow common audio types)
        content_type = head_resp.headers.get("content-type", "").lower()
        if content_type and not any(ct in content_type for ct in ["audio/", "video/", "application/octet-stream"]):
            logger.warning(f"Suspicious content type for URL {url}: {content_type}")
    except requests.RequestException as e:
        logger.warning(f"HEAD request failed for {url}: {e}, proceeding with download")
    
    raw_path = os.path.join(tmp_dir, "in.bin")
    wav_path = os.path.join(tmp_dir, "in.wav")

    with requests.get(url, stream=True, timeout=30) as r:
        r.raise_for_status()
        total = 0
        with open(raw_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                if not chunk:
                    continue
                total += len(chunk)
                if total > MAX_DOWNLOAD_BYTES:
                    raise ValueError("Remote file exceeds size limit (50 MB).")
                f.write(chunk)

    # Normalize via ffmpeg → 22.05kHz mono WAV
    # You can tweak the sample rate/channels to match your pipeline defaults.
    subprocess.run(
        ["ffmpeg", "-y", "-i", raw_path, "-ar", str(TARGET_SR), "-ac", "1", wav_path],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return wav_path

def _process_direct_url_job(url: str) -> dict:
    """
    Background worker entrypoint. Mirrors the shape returned by /transcribe (file upload).
    """
    tmp_dir = tempfile.mkdtemp(prefix="url_")
    try:
        audio_path = _download_to_wav(tmp_dir, url)
        result = process_audio_file_sync(audio_path)  # <-- your existing sync processor
        tempo = result.get("tempo")
        # Try to derive a nice filename from the URL path
        filename = _sanitize_filename(Path(url).name)
        return {
            "metadata": {
                "filename": filename,
                **result.get("metadata", {}),
            },
            "chords": result.get("chords", []),
            "melody": result.get("melody", []),
            "tempo": tempo if isinstance(tempo, dict) else ({"bpm": tempo} if tempo else None),
        }
    finally:
        try:
            import shutil
            shutil.rmtree(tmp_dir, ignore_errors=True)
        except OSError:
            pass

# --- FastAPI app and lifespan ---


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Create a thread pool executor for blocking work
    from concurrent.futures import ThreadPoolExecutor
    app.state.executor = ThreadPoolExecutor(max_workers=config.MAX_WORKERS)
    yield
    # Cleanup executor on shutdown
    try:
        app.state.executor.shutdown(wait=False)
    except Exception:
        pass


APP_ENV = os.getenv("APP_ENV", "production").lower()
IS_DEV = APP_ENV in ("dev", "development", "local")
app = FastAPI(
    title="Guitar Music Helper Backend",
    description="Audio transcription API for guitar music helper",
    version="1.0.0",
    docs_url="/docs" if config.is_development else None,
    redoc_url="/redoc" if config.is_development else None,
    lifespan=lifespan,
)

# Initialize Redis/RQ queue for background audio jobs (optional)
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
try:
    redis_pool = ConnectionPool.from_url(REDIS_URL, max_connections=10)
    redis_conn = Redis(connection_pool=redis_pool)
    job_queue = Queue("gmh-audio", connection=redis_conn)
    logger.info("RQ connected and queue ready.")
except Exception as _e:
    redis_conn = None
    job_queue = None
    logger.warning(f"RQ/Redis not available: {_e}")



@app.post("/transcribe-youtube", summary="Submit a YouTube URL for background transcription", tags=["Transcription"])
@limiter.limit("3/minute")
async def transcribe_youtube(request: Request, payload: YouTubeJob):
    if not job_queue:
        return JSONResponse(status_code=503, content={"success": False, "error": "Background queue not available"})
    # Ensure URL sanity
    if not is_youtube_url(payload.url):
        return JSONResponse(status_code=400, content={"success": False, "error": "Invalid YouTube URL"})

    # Dependencies check
    missing = _check_youtube_dependencies()
    if missing:
        return JSONResponse(status_code=503, content={"success": False, "error": "Missing system dependencies", "details": missing})

    # Retry transient failures up to 3 times with increasing intervals
    retry_policy = Retry(max=3, interval=[10, 30, 60])
    job = job_queue.enqueue(process_youtube_job, payload.url, retry=retry_policy)
    return JSONResponse(status_code=202, content={"success": True, "job_id": job.get_id()})

# Alias to support existing frontend integrations: /transcribe/youtube
@app.post("/transcribe/youtube", summary="(alias) Submit a YouTube URL", tags=["Transcription"])
@limiter.limit("3/minute")
async def transcribe_youtube_alias(request: Request, payload: YouTubeJob):
    return await transcribe_youtube(request, payload)


@app.post("/transcribe-url", summary="Submit a direct audio URL for background transcription", tags=["Transcription"])
@limiter.limit("3/minute")
async def transcribe_url(request: Request, payload: DirectURLJob):
    if not job_queue:
        return JSONResponse(status_code=503, content={"success": False, "error": "Background queue not available"})
    if not _is_http_url(payload.url):
        return JSONResponse(status_code=400, content={"success": False, "error": "Invalid or non-HTTP URL"})
    try:
        # Enqueue a background job; worker must be running: rq worker -u "$REDIS_URL" gmh-audio
        retry_policy = Retry(max=3, interval=[10, 30, 60])
        job = job_queue.enqueue(_process_direct_url_job, payload.url, retry=retry_policy)
        return JSONResponse(status_code=202, content={"success": True, "job_id": job.get_id()})
    except Exception as e:
        return JSONResponse(status_code=500, content={"success": False, "error": str(e)})


@app.get("/jobs/{job_id}", summary="Check background transcription job status", tags=["Transcription"])
@limiter.limit("120/minute", key_func=job_rate_key)
async def job_status(request: Request, job_id: str):
    if not redis_conn:
        return JSONResponse(status_code=503, content={"job_id": job_id, "status": "unavailable", "finished": False, "result": None, "error": "Queue not available"})
    try:
        from rq.job import Job
        job = Job.fetch(job_id, connection=redis_conn)
    except Exception:
        return JSONResponse(status_code=404, content={"job_id": job_id, "status": "not_found", "finished": False, "result": None, "error": "Unknown job id"})
    return JSONResponse(status_code=200, content=_job_payload(job, include_result=False))

# Alias endpoint that returns the result directly when finished (common in UIs)
@app.get("/jobs/{job_id}/result", summary="Get finished job result", tags=["Transcription"])
@limiter.limit("120/minute", key_func=job_rate_key)
async def job_result(request: Request, job_id: str):
    if not redis_conn:
        return JSONResponse(status_code=503, content={"job_id": job_id, "status": "unavailable", "finished": False, "result": None, "error": "Queue not available"})
    try:
        from rq.job import Job
        job = Job.fetch(job_id, connection=redis_conn)
    except Exception:
        return JSONResponse(status_code=404, content={"job_id": job_id, "status": "not_found", "finished": False, "result": None, "error": "Unknown job id"})
    payload = _job_payload(job, include_result=True)
    if job.is_finished:
        return JSONResponse(status_code=200, content=payload)
    if job.is_failed:
        return JSONResponse(status_code=500, content=payload)
    return JSONResponse(status_code=202, content=payload)

# Configure rate limiting
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
app.add_middleware(SlowAPIMiddleware)

# -----------------------------
# Helpers: deps + job formatting
# -----------------------------
def _check_youtube_dependencies() -> list[str]:
    """Return a list of missing tools required for YouTube transcription."""
    missing: list[str] = []
    if _which("ffmpeg") is None:
        missing.append("ffmpeg is not installed or not on PATH. Install ffmpeg in the Docker image.")
    if _importlib_util.find_spec("yt_dlp") is None:
        missing.append("Python package 'yt-dlp' is missing. Add it to requirements.txt (e.g. yt-dlp==2025.6.30).")
    return missing

def _job_payload(job, include_result: bool = True) -> dict:
    """Consistent JSON shape for job status/results (rq-version-safe)."""
    try:
        status = job.get_status(refresh=False) or "unknown"
    except Exception:
        # Fallback for very old/new RQ versions
        status = (
            "finished" if getattr(job, "is_finished", False) else
            "failed"   if getattr(job, "is_failed", False)   else
            "unknown"
        )
    return {
        "job_id": job.get_id(),
        "status": status,
        "finished": status == "finished",
        "result": (job.result if (include_result and status == "finished") else None),
        "error": (str(job.exc_info) if status == "failed" else None),
    }

# Initialize metrics collection
app.state.metrics = SimpleMetrics()

# --- SimpleTTLCache import (fallback if not available as a module) ---
try:
    from simple_ttl_cache import SimpleTTLCache
except ImportError:
    import threading, time
    class SimpleTTLCache:
        def __init__(self, maxsize=50, ttl=3600):
            self.maxsize = maxsize
            self.ttl = ttl
            self._cache = {}
            self._lock = threading.Lock()
        def get(self, key):
            with self._lock:
                v = self._cache.get(key)
                if v is None:
                    return None
                value, expires = v
                if expires < time.time():
                    del self._cache[key]
                    return None
                return value
        def set(self, key, value):
            with self._lock:
                if len(self._cache) >= self.maxsize:
                    # Remove oldest
                    oldest = min(self._cache.items(), key=lambda item: item[1][1])[0]
                    del self._cache[oldest]
                self._cache[key] = (value, time.time() + self.ttl)
app.state.cache = SimpleTTLCache(maxsize=config.CACHE_MAX_ITEMS, ttl=config.CACHE_TTL_SECONDS)

# --- Request Size Limit Middleware ---
@app.middleware("http")
async def limit_request_size(request: Request, call_next):
    cl = request.headers.get("content-length")
    if cl:
        try:
            size = int(cl)
            # allow ~1MB form overhead
            if size > (config.MAX_FILE_SIZE + 1024 * 1024):
                return JSONResponse(status_code=413, content={"error": "Request too large"})
        except ValueError:
            pass
    return await call_next(request)

# --- Request ID Middleware ---
@app.middleware("http")
async def add_request_id(request: Request, call_next):
    """Add unique request ID for tracing"""
    request_id = str(uuid.uuid4())
    request.state.request_id = request_id
    token = request_context.set({
        "request_id": request_id,
        "method": request.method,
        "path": request.url.path,
        "client_ip": get_remote_address(request)
    })
    try:
        response = await call_next(request)
    finally:
        request_context.reset(token)
    response.headers["X-Request-ID"] = request_id
    return response

# --- CORS Middleware Configuration ---
# Configure CORS with environment-based origins for security
logger.info(f"🌐 CORS configured for {config.ENVIRONMENT} environment")

# Simplified CORS origins list - remove trailing slashes and duplicates
origins = [
    "http://localhost:5173",
    "http://127.0.0.1:5173", 
    "http://localhost:5174",
    "http://127.0.0.1:5174",
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "https://guitar-music-helper.vercel.app",
    "https://web-production-84b20.up.railway.app",
    # Add current Vercel preview domains
    "https://guitar-music-helper-7g9fwzvch-dberzons-projects.vercel.app",
    "https://guitar-music-helper-h7erfkcq4-dberzons-projects.vercel.app",
    "https://guitar-music-helper-h9i35sgkq-dberzons-projects.vercel.app"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_origin_regex=r"^https://.*-dberzon-projects\.vercel\.app$",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["X-Request-ID", "X-RateLimit-Remaining", "X-RateLimit-Reset"],
    max_age=600,
)


# Fallback middleware: ensure Access-Control-Allow-Origin is always present on responses.
# This helps when an exception handler or other path returns a response without CORS headers
# (some hosting environments may short-circuit middleware in error paths).
@app.middleware("http")
async def ensure_cors_header(request: Request, call_next):
    # Get the request origin
    origin = request.headers.get("origin")
    
    try:
        response = await call_next(request)
    except Exception as e:
        # If an unhandled error occurs, create a basic error response with CORS headers
        from fastapi.responses import JSONResponse
        response = JSONResponse(
            status_code=500,
            content={"success": False, "error": "Internal server error"}
        )
    
    # Always ensure CORS headers are present
    if origin:
        # Check if origin is allowed
        allowed = False
        for allowed_origin in config.CORS_ORIGINS:
            if origin == allowed_origin:
                response.headers["Access-Control-Allow-Origin"] = origin
                allowed = True
                break
        
        # Check regex pattern if not already allowed
        if not allowed and config.CORS_ORIGIN_REGEX:
            import re
            if re.match(config.CORS_ORIGIN_REGEX, origin):
                response.headers["Access-Control-Allow-Origin"] = origin
                allowed = True
        
        # If not allowed but in development, be permissive
        if not allowed and config.is_development:
            response.headers["Access-Control-Allow-Origin"] = origin
    else:
        # No origin header, set to first allowed origin or wildcard
        origins = config.CORS_ORIGINS or []
        response.headers["Access-Control-Allow-Origin"] = origins[0] if origins else "*"
    
    # Always add other CORS headers for preflight compatibility
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    response.headers["Access-Control-Allow-Headers"] = "*"
    response.headers["Access-Control-Allow-Credentials"] = "true"
    
    return response

# --- Exception Handlers ---
# These handlers ensure that clients always receive a consistent,
# structured JSON error response.

def add_cors_headers_to_response(response: JSONResponse, request: Request):
    """Helper function to add CORS headers to any response"""
    origin = request.headers.get("origin")
    
    if origin:
        # Check if origin is allowed
        allowed = False
        for allowed_origin in config.CORS_ORIGINS:
            if origin == allowed_origin:
                response.headers["Access-Control-Allow-Origin"] = origin
                allowed = True
                break
        
        # Check regex pattern if not already allowed
        if not allowed and config.CORS_ORIGIN_REGEX:
            import re
            if re.match(config.CORS_ORIGIN_REGEX, origin):
                response.headers["Access-Control-Allow-Origin"] = origin
                allowed = True
        
        # If not allowed but in development, be permissive
        if not allowed and config.is_development:
            response.headers["Access-Control-Allow-Origin"] = origin
    else:
        # No origin header, set to first allowed origin or wildcard
        origins = config.CORS_ORIGINS or []
        response.headers["Access-Control-Allow-Origin"] = origins[0] if origins else "*"
    
    # Always add other CORS headers
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    response.headers["Access-Control-Allow-Headers"] = "*"
    response.headers["Access-Control-Allow-Credentials"] = "true"
    return response

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Handles any exception not caught by more specific handlers."""
    logger.error(f"Unhandled exception for request {request.url}: {exc}", exc_info=True)
    response = JSONResponse(
        status_code=500,
        content={
            "success": False,
            "error": {
                "code": "INTERNAL_SERVER_ERROR",
                "message": "An unexpected internal server error occurred.",
            },
        },
    )
    return add_cors_headers_to_response(response, request)

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Handles FastAPI's built-in HTTPExceptions."""
    response = JSONResponse(
        status_code=exc.status_code,
        content={
            "success": False,
            "error": {
                "code": "REQUEST_ERROR",
                "message": exc.detail,
            }
        },
        headers=exc.headers,
    )
    return add_cors_headers_to_response(response, request)

@app.exception_handler(AudioProcessingError)
async def audio_processing_error_handler(request: Request, exc: AudioProcessingError):
    """Handles errors specific to the audio processing pipeline."""
    request_id = getattr(request.state, 'request_id', None)
    logger.warning(f"Audio processing error for request {request.url} (ID: {request_id}): {exc}")
    # Map timeouts to a more appropriate HTTP status
    status_code = 504 if isinstance(exc, ProcessingTimeoutError) else 422
    code = "TIMEOUT" if isinstance(exc, ProcessingTimeoutError) else "AUDIO_PROCESSING_ERROR"
    response = JSONResponse(
        status_code=status_code,
        content={
            "success": False,
            "error": {
                "code": code,
                "message": "Processing timed out." if status_code == 504 else "Failed to process the provided audio file.",
                "details": str(exc),
                "request_id": request_id
            },
        },
    )
    return add_cors_headers_to_response(response, request)

@app.exception_handler(DependencyError)
async def dependency_error_handler(request: Request, exc: DependencyError):
    """Handles errors when required libraries are not loaded."""
    request_id = getattr(request.state, 'request_id', None)
    logger.error(f"Dependency error for request {request.url} (ID: {request_id}): {exc}")
    response = JSONResponse(
        status_code=503,  # Service Unavailable
        content={
            "success": False,
            "error": {
                "code": "SERVICE_UNAVAILABLE",
                "message": "The transcription service is temporarily unavailable.",
                "details": str(exc),
                "request_id": request_id
            },
        },
    )
    return add_cors_headers_to_response(response, request)
# --- Dependency Injection & Validation Helpers ---

def get_metrics_collector(request: Request) -> SimpleMetrics:
    """Dependency to get the metrics collector instance."""
    return request.app.state.metrics

def check_dependencies():
    """Ensure required ML deps are loaded. Pydantic models are optional since we return plain dicts."""
    if not DEPENDENCIES_LOADED:
        raise DependencyError("Required ML dependencies are not loaded.")

def validate_file(file: UploadFile = File(...)) -> UploadFile:
    """
    Validates the uploaded file's existence, size, and extension.
    This runs as a dependency, cleaning up the endpoint logic.
    """
    if not file or not file.filename:
        raise HTTPException(status_code=400, detail="No file provided.")

    # NOTE: Starlette's UploadFile has no reliable .size; enforce size after saving.

    # Sanitize filename (strip any path) and block dotfiles & double extensions
    safe_name = Path(file.filename).name
    if not safe_name or safe_name.startswith("."):
        raise HTTPException(status_code=400, detail="Invalid filename.")
    if safe_name.count(".") > 1:
        # Allow benign multi-dot names, block suspicious inner extensions
        dangerous = {"exe","js","php","sh","bat","cmd","com","scr"}
        inner_parts = [p.lower() for p in safe_name.split(".")[:-1]]
        if any(p in dangerous for p in inner_parts):
            raise HTTPException(status_code=400, detail="Suspicious multi-extension filename.")

    file_extension = Path(safe_name).suffix.lower()
    if file_extension not in config.ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=415,  # Unsupported Media Type
            detail=f"Unsupported file type '{file_extension}'. Allowed types are: {', '.join(config.ALLOWED_EXTENSIONS)}"
        )
    
    # Validate MIME type matches the file extension - allow common variations
    allowed_types = ALLOWED_MIME_TYPES.get(file_extension, [])
    if allowed_types and file.content_type not in allowed_types:
        raise HTTPException(
            status_code=415, 
            detail=f"MIME type {file.content_type} not allowed for {file_extension} files. Allowed: {', '.join(allowed_types)}"
        )
    
    return file

def validate_audio_content(tmp_path: str) -> bool:
    """Optional content sniff (best-effort). Returns True if looks like audio."""
    try:
        import magic  # python-magic; optional, will skip if missing
        mime = magic.from_file(tmp_path, mime=True)
        return bool(mime and mime.startswith("audio/"))
    except (ImportError, ModuleNotFoundError):
        # magic is optional; don't block when missing
        return True
    except Exception:
        # If magic errors for other reasons, be permissive
        return True

@app.get("/", summary="API Root", tags=["Status"])
async def root():
    """Provides basic service information and status."""
    return {
        "service": "Guitar Music Helper Audio Transcription API",
        "version": app.version,
        "status": "healthy" if DEPENDENCIES_LOADED and MODELS_LOADED else "degraded",
        "environment": config.ENVIRONMENT,
        "backend_url": config.BACKEND_URL,
    }

@app.get("/environment", summary="Environment Information", tags=["Status"])
async def environment_info():
    """Provides environment configuration information for frontend connection."""
    return {
        "environment": config.ENVIRONMENT,
        "backend_url": config.BACKEND_URL,
        "is_development": config.is_development,
        "is_production": config.is_production,
        "cors_origins": config.CORS_ORIGINS,
        "debug_endpoints_enabled": config.ENABLE_DEBUG_ENDPOINTS,
    }

# --- Health check helper functions ---
async def check_redis_connection() -> bool:
    try:
        await asyncio.get_event_loop().run_in_executor(None, redis_conn.ping)
        return True
    except Exception as e:
        logger.warning("Redis ping failed: %s", e)
        return False

def check_disk_space(min_free_bytes: int = 300 * 1024 * 1024) -> bool:
    import shutil
    total, used, free = shutil.disk_usage("/tmp")
    return free >= min_free_bytes

def check_ffmpeg() -> bool:
    try:
        _run(["ffmpeg", "-version"], "ffmpeg")
        _run(["ffprobe", "-version"], "ffprobe")
        return True
    except Exception:
        return False

@app.get("/health", summary="Health Check", tags=["Status"])
async def health_check():
    """Performs a detailed health check of the service and its dependencies."""
    try:
        return {
            "status": "healthy" if DEPENDENCIES_LOADED and MODELS_LOADED else "degraded",
            "dependencies_loaded": DEPENDENCIES_LOADED,
            "models_loaded": MODELS_LOADED,
            "supported_formats": list(config.ALLOWED_EXTENSIONS) if config.ALLOWED_EXTENSIONS else [],
            "max_file_size_mb": config.MAX_FILE_SIZE_MB if config.MAX_FILE_SIZE_MB is not None else 0,
            "timestamp": time.time(),
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.get("/health/live", summary="Liveness Probe", tags=["Status"])
def liveness_probe():
    return {"alive": True}

@app.get("/health/ready", summary="Readiness Probe", tags=["Status"])
async def readiness_probe():
    checks = {
        "api": True,
        "redis": bool(redis_conn) and bool(job_queue),
        "disk_space": check_disk_space(),
        "ffmpeg": _which("ffmpeg") is not None,
        "yt_dlp": _find_spec("yt_dlp") is not None,
    }
    return {"ready": all(checks.values()), "checks": checks}

@app.get("/health/detailed", summary="Detailed Health Check", tags=["Status"])
async def detailed_health_check():
    """More comprehensive health check including system resources"""
    checks = {
        "api": "healthy",
        "dependencies": DEPENDENCIES_LOADED,
        "models": MODELS_LOADED,
        "memory": {
            "used_mb": get_memory_usage(),
            "psutil_available": PSUTIL_AVAILABLE
        },
        "executor": {
            "max_workers": config.MAX_WORKERS
        }
    }
    
    # Add more detailed memory info if psutil is available
    if PSUTIL_AVAILABLE:
        try:
            mem = psutil.virtual_memory()
            disk = psutil.disk_usage(tempfile.gettempdir())
            checks["memory"].update({
                "available_mb": mem.available / 1024 / 1024,
                "percent_used": mem.percent
            })
            checks["disk"] = {
                "temp_dir_free_gb": disk.free / 1024**3
            }
        except Exception as e:
            checks["memory"]["error"] = str(e)
    
    # Determine overall health
    status = "healthy"
    if not DEPENDENCIES_LOADED or not MODELS_LOADED:
        status = "degraded"
    elif PSUTIL_AVAILABLE and checks["memory"].get("percent_used", 0) > 90:
        status = "warning"
    
    return {"status": status, "checks": checks}

# --- Audio Processing Helper Functions ---
# Note: Heavy ML libraries (librosa, basic_pitch, numpy) are imported lazily
# inside functions to reduce startup memory footprint and improve cold-start performance

# capture helper (stdout) for lightweight probes like ffprobe - keep alongside _run
def _capture(cmd: list[str], name: str, timeout: int = 30) -> str:
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    if proc.returncode != 0:
        raise RuntimeError(f"{name} failed ({proc.returncode}): {proc.stderr.strip()}")
    return (proc.stdout or "").strip()

def _run(cmd, step_name):
    """Run subprocess command with error handling"""
    try:
        subprocess.run(cmd, check=True, capture_output=True)
    except subprocess.CalledProcessError as e:
        # e.stderr may be bytes or None; guard decode safely
        err = (e.stderr.decode("utf-8", "ignore") if isinstance(e.stderr, (bytes, bytearray)) else str(e.stderr))
        raise RuntimeError(f"{step_name} failed: {err}")

def _ffmpeg_basic_wav(input_path: str, output_path: str, sr: int = TARGET_SR, max_sec: int = MAX_AUDIO_DURATION_S) -> None:
    """
    Downmix to mono, resample, **trim to max_sec** at the demuxer, and write wav.
    Trimming here prevents Python from loading huge arrays into memory.
    """
    # -t applies an output duration limit; keep it early in the chain.
    cmd = [
        "ffmpeg", "-y",
        "-i", input_path,
        "-t", str(max_sec),
        "-ac", "1",              # mono
        "-ar", str(sr),          # sample rate
        "-vn", "-sn", "-dn",     # no video/subs/data
        output_path,
    ]
    _run(cmd, "ffmpeg")

def _is_audio_by_ffprobe(path: str) -> bool:
    """
    Validate that the file has at least one audio stream before loading into Python.
    Avoids non-audio uploads causing decoder churn.
    """
    try:
        out = _capture(
            ["ffprobe", "-v", "error", "-select_streams", "a:0",
             "-show_entries", "stream=codec_type", "-of", "csv=p=0", path],
            "ffprobe"
        )
        return "audio" in out.lower()
    except Exception as e:
        logger.warning("ffprobe validation failed: %s", e)
        return False

async def _prepare_audio(input_path: str) -> str:
    """
    Convert arbitrary media to analysis-ready wav (mono, resampled).
    Returns a path to a temp wav file.
    """
    fd, wav_path = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    _ffmpeg_basic_wav(input_path, wav_path, sr=TARGET_SR, max_sec=MAX_AUDIO_DURATION_S)
    return wav_path

def _load_audio_array(wav_path: str):
    """
    Load into memory for model. Keep dtype small to reduce RAM.
    Assumes wav_path already trimmed to MAX_AUDIO_DURATION_S.
    """
    import soundfile as sf
    audio, sr = sf.read(wav_path, dtype="float32", always_2d=False)
    # Safety: if something upstream failed to trim, enforce here too
    # (fast slice; avoids unexpected spikes).
    import math
    max_samples = int(MAX_AUDIO_DURATION_S * sr)
    if audio.shape[0] > max_samples:
        audio = audio[:max_samples]
    return audio, sr

def load_audio_file(file_path: str, max_duration: Optional[float] = None) -> tuple:
    """Load audio file with memory-efficient settings"""
    try:
        import librosa
        # use the configured TARGET_SR for consistency across all ingest paths
        audio, sr = librosa.load(file_path, sr=TARGET_SR, mono=True, duration=max_duration)
        duration = librosa.get_duration(y=audio, sr=sr)
        logger.info(f"Audio loaded: duration={duration:.2f}s, sample_rate={sr}Hz, audio_shape={audio.shape}")
        return audio, sr, duration
    except Exception as e:
        logger.error(f"Failed to load audio file {file_path}: {e}")
        raise

def run_basic_pitch_prediction(file_path: str) -> tuple:
    """Run basic-pitch prediction on audio file"""
    try:
        from basic_pitch.inference import predict
        model_output, midi_data, note_events = predict(file_path)
        logger.info(f"Basic-pitch prediction complete. Found {len(note_events)} note events.")
        return model_output, midi_data, note_events
    except Exception as e:
        logger.error(f"Basic-pitch prediction failed for {file_path}: {e}")
        raise

# --- Naive chord fallback (Basic-Pitch‑friendly) -----------------------------
def get_naive_chords_from_notes(events: List[dict], duration: float, window: float = 1.0) -> List[Dict]:
    """
    Fallback chord detection tuned to Basic-Pitch note events.
    Looks at active notes in fixed windows, selects a root by mode,
    and labels crude maj/min triads. Merges consecutive duplicates.
    """
    # Improved fallback chord detection: shorter window, note duration weighting
    try:
        import numpy as np
    except Exception:
        return []

    if not events or duration <= 0:
        return []

    pc_names = ["C","C#","D","D#","E","F","F#","G","G#","A","A#","B"]
    chords_out: List[Dict] = []
    t = 0.0

    def _unpack_event(ev) -> tuple[float, float, int]:
        """Return (start, end, midi) from either a dict-like event or a tuple/list event."""
        if isinstance(ev, (list, tuple)):
            # basic-pitch note tuple: (start, end, pitch, amplitude, ...)
            try:
                st = float(ev[0])
                en = float(ev[1])
                midi = int(ev[2])
            except Exception:
                return 0.0, 0.0, 60
            return st, en, midi
        # dict-like event
        try:
            st = float(ev.get("start_time") or ev.get("onset_time") or 0.0)
            en = float(ev.get("end_time") or ev.get("offset_time") or 0.0)
            midi = int(ev.get("midi_note") or ev.get("pitch") or 60)
        except Exception:
            return 0.0, 0.0, 60
        return st, en, midi

    # window param is now respected from caller
    while t < max(duration, 1e-3):
        t_end = min(t + window, duration)
        # Weight pitch classes by note duration in window
        pcs: List[int] = []
        weights: List[float] = []
        for ev in events:
            st, en, midi = _unpack_event(ev)
            pc = midi % 12
            overlap = max(0.0, min(t_end, en) - max(t, st))
            if overlap > 0:
                pcs.append(pc)
                weights.append(overlap)

        if pcs:
            hist = np.zeros(12)
            for pc, w in zip(pcs, weights):
                hist[pc] += w
            root = int(hist.argmax())
            has_m3 = hist[(root + 3) % 12]
            has_M3 = hist[(root + 4) % 12]
            has_P5 = hist[(root + 7) % 12]
            qual = "" if (has_M3 >= has_m3 and has_P5) else "m"
            name = pc_names[root] + qual
            chords_out.append({"time": float(t), "duration": float(t_end - t), "chord": name})

        t = t_end

    # merge consecutive duplicates
    merged: List[Dict] = []
    for c in chords_out:
        if merged and merged[-1]["chord"] == c["chord"]:
            # Extend previous segment
            merged[-1]["duration"] += c["duration"]
        else:
            merged.append(c)
    return merged


def smooth_chords(chords: List[Dict]) -> List[Dict]:
    """Merge consecutive chord segments that have the same chord label.

    This normalizes output from Madmom or fallback chorders by combining
    adjacent segments with identical chord names into single longer segments.
    """
    if not chords:
        return []
    smoothed: List[Dict] = []
    for chord in chords:
        if smoothed and chord.get("chord") == smoothed[-1].get("chord"):
            # Extend previous segment
            smoothed[-1]["duration"] += chord.get("duration", 0)
        else:
            smoothed.append(dict(chord))
    return smoothed

# --- Synchronous Processing Function ---

def process_audio_file_sync(tmp_path: str, debug: bool = False) -> ProcessingResult:
    """
    Orchestrator function that coordinates audio processing steps.
    Broken down into smaller, testable functions for better maintainability.
    """
    # heavy-lift transcription and harmony
    import gc  # Import garbage collector for memory management
    
    try:
        logger.info(f"Starting audio processing at path: {tmp_path}")
        # Check file size before processing
        file_size = os.path.getsize(tmp_path)
        logger.info(f"File size: {file_size / (1024*1024):.2f} MB")
        # Force garbage collection before starting
        gc.collect()
        # Load audio file
        audio, sr, duration = load_audio_file(tmp_path)
        logger.info(f"Loaded audio: duration={duration:.2f}s, sample_rate={sr}Hz, shape={getattr(audio, 'shape', None)}")
        note_count = None
        first_notes = None
        # Run basic-pitch prediction
        model_output, midi_data, note_events = run_basic_pitch_prediction(tmp_path)
        note_count = len(note_events) if note_events is not None else 0
        first_notes = note_events[:5] if note_events else []
        # Process the output
        logger.info("Processing basic-pitch output...")
        transcription_data = process_basic_pitch_output(
            model_output, midi_data, note_events, sr, duration
        )
        logger.info("Processing complete, returning results...")
        # Madmom chord recognition
        madmom_chords = detect_chords_madmom(tmp_path)
        # Smoothing for Madmom chords (merge consecutive same-chord segments)
        madmom_chords_smoothed = smooth_chords(madmom_chords)
        chords_list = madmom_chords_smoothed if madmom_chords_smoothed else transcription_data.get("chords", [])
        fallback_chords = None
        if not chords_list:
            logger.warning("No chords from Madmom or process_basic_pitch_output, using improved fallback chorder (window=0.4s, duration-weighted)")
            chords_list = get_naive_chords_from_notes(note_events, duration, window=0.4)
            fallback_chords = chords_list
        else:
            fallback_chords = get_naive_chords_from_notes(note_events, duration, window=0.4)
        # Clean up intermediate data after potential fallback
        del model_output, midi_data, note_events, audio
        gc.collect()
        # Return a dictionary with the core results
        result = {
            "metadata": {"duration": duration, "sampleRate": sr},
            "chords": chords_list,
            "melody": transcription_data.get("melody", []),
            "tempo": transcription_data.get("tempo"),
        }
        # Optionally add debug info if requested (see /transcribe endpoint)
        if debug:
            result["debug"] = {
                "analysis_sr": sr,
                "downmixed": True,
                "note_count": note_count,
                "first_note_events": first_notes,
                "madmom_chords": madmom_chords,
                "fallback_chords": fallback_chords,
            }
        return result
    except Exception as e:
        logger.error(f"Core audio processing failed: {e}", exc_info=True)
        gc.collect()
        raise AudioProcessingError(f"Processing failed: {e}") from e

@app.post("/test-minimal-processing", 
    summary="Test Minimal Audio Processing", 
    tags=["Diagnostics"],
    include_in_schema=config.ENABLE_DEBUG_ENDPOINTS
)
@limiter.limit("3/minute")
async def test_minimal_processing(request: Request, file: UploadFile = Depends(validate_file)):
    """
    Test minimal audio processing to identify where exactly the failure occurs.
    """
    require_debug_enabled()
    
    start_time = time.time()
    tmp_path = None
    
    try:
        if not DEPENDENCIES_LOADED:
            return {"success": False, "error": "Dependencies not loaded"}
        
        # Save file
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(file.filename).suffix) as tmp_file:
            tmp_path = tmp_file.name
        
        await file.seek(0)
        async with aiofiles.open(tmp_path, 'wb') as out_file:
            while content := await file.read(1024 * 1024):
                await out_file.write(content)
        
        file_size = os.path.getsize(tmp_path)
        step_results = {"file_upload": f"OK - {file_size / (1024*1024):.2f} MB"}
        
        # Test librosa loading
        try:
            import librosa
            import gc
            gc.collect()  # Clean memory before loading
            
            audio, sr = librosa.load(tmp_path, sr=TARGET_SR, mono=True, duration=10.0)  # Limit to 10 seconds
            step_results["librosa_load"] = f"OK - {len(audio)} samples at {sr}Hz"
            
            # Clean up audio data immediately
            del audio
            gc.collect()
            
        except Exception as e:
            step_results["librosa_load"] = f"ERROR: {e}"
            return {"success": False, "step_results": step_results, "failed_at": "librosa_load"}
        
        # Test basic-pitch prediction (this is likely where it fails)
        try:
            from basic_pitch.inference import predict
            
            # Try to predict - this might cause OOM
            model_output, midi_data, note_events = predict(tmp_path)
            step_results["basic_pitch_predict"] = f"OK - {len(note_events)} note events"
            
            # Clean up immediately
            del model_output, midi_data, note_events
            gc.collect()
            
        except Exception as e:
            step_results["basic_pitch_predict"] = f"ERROR: {e}"
            return {"success": False, "step_results": step_results, "failed_at": "basic_pitch_predict", "error": str(e)}
        
        processing_time = time.time() - start_time
        return {
            "success": True,
            "step_results": step_results,
            "processing_time": round(processing_time, 2),
            "message": "All steps completed successfully"
        }
        
    except Exception as e:
        logger.error(f"Error in minimal processing test: {e}", exc_info=True)
        return {"success": False, "error": str(e)}
        
    finally:
        # Clean up temp file
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
            except:
                pass

@app.post("/test-dependencies", 
    summary="Test ML Dependencies", 
    tags=["Diagnostics"],
    include_in_schema=config.ENABLE_DEBUG_ENDPOINTS
)
@limiter.limit("5/minute")
async def test_dependencies(request: Request):
    """
    Test if ML dependencies can be imported and used without file processing.
    """
    require_debug_enabled()
    
    try:
        if not DEPENDENCIES_LOADED:
            return {"success": False, "error": "Dependencies not loaded"}
        
        # Test basic imports
        test_results = {
            "librosa": "unknown",
            "numpy": "unknown", 
            "basic_pitch": "unknown"
        }
        
        # Test librosa
        try:
            import librosa
            test_results["librosa"] = f"OK - version {librosa.__version__}"
        except Exception as e:
            test_results["librosa"] = f"ERROR: {e}"
        
        # Test numpy
        try:
            import numpy as np
            test_results["numpy"] = f"OK - version {np.__version__}"
        except Exception as e:
            test_results["numpy"] = f"ERROR: {e}"
        
        # Test basic-pitch predict function
        try:
            from basic_pitch.inference import predict
            test_results["basic_pitch"] = "OK - predict function imported"
        except Exception as e:
            test_results["basic_pitch"] = f"ERROR: {e}"
        
        return {
            "success": True,
            "test_results": test_results,
            "dependencies_loaded": DEPENDENCIES_LOADED,
            "models_loaded": MODELS_LOADED
        }
        
    except Exception as e:
        logger.error(f"Error testing dependencies: {e}", exc_info=True)
        return {
            "success": False,
            "error": str(e)
        }

@app.post("/transcribe-status", summary="Check Transcription Capability", tags=["Transcription"])
@limiter.limit("10/minute")
async def transcribe_status(request: Request, file: UploadFile = Depends(validate_file)):
    """
    Test endpoint to check if transcription would be possible without actually processing.
    Returns estimated memory requirements and processing feasibility.
    """
    try:
        # Roughly infer file size from Content-Length (multipart adds overhead)
        file_size_mb = None
        cl = request.headers.get("content-length")
        if cl:
            try:
                content_len = int(cl)
                # subtract ~256KB to offset multipart form overhead (heuristic)
                approx_bytes = max(0, content_len - 256 * 1024)
                file_size_mb = approx_bytes / (1024 * 1024)
            except ValueError:
                pass

        # Basic-pitch typically needs 3–5× the audio size; use 4× as a conservative middle
        estimated_memory_mb = (file_size_mb * 4) if file_size_mb is not None else 50
        
        # Allow overriding memory limit via env if you're on a paid plan
        railway_memory_limit = int(os.getenv("RAILWAY_RAM_MB", "8192"))
        
        return {
            "success": True,
            "file_info": {
                "filename": file.filename,
                "size_mb": round(file_size_mb, 2) if file_size_mb is not None else None,
                "format": Path(file.filename).suffix.lower()
            },
            "memory_analysis": {
                "estimated_memory_needed_mb": round(estimated_memory_mb, 2),
                "railway_memory_limit_mb": railway_memory_limit,
                "feasible": estimated_memory_mb < railway_memory_limit,
                "recommendation": "File too large for current Railway plan" if estimated_memory_mb >= railway_memory_limit else "Processing should be feasible"
            },
            "dependencies": {
                "dependencies_loaded": DEPENDENCIES_LOADED,
                "models_loaded": MODELS_LOADED
            }
        }
        
    except Exception as e:
        logger.error(f"Error in transcribe-status: {e}")
        return {
            "success": False,
            "error": str(e)
        }

@app.post(
    "/transcribe",
    summary="Transcribe Audio File",
    tags=["Transcription"],
    response_model=TranscribeResponse,
    response_model_exclude_none=True,
    dependencies=[Depends(check_dependencies)], # Protects the endpoint if dependencies are missing
)
@limiter.limit("5/minute")
async def transcribe_audio(
    request: Request,
    file: UploadFile = Depends(validate_file),
    metrics: SimpleMetrics = Depends(get_metrics_collector),
):
    """
    Accepts an audio file, transcribes it to find chords and melody,
    and returns the structured data. Optional query param: debug=true to include extra analysis info.
    """
    import time, os
    start_time = time.time()
    endpoint_name = "transcribe"
    metrics.record_request(endpoint_name)
    debug = request.query_params.get("debug", "false").lower() == "true"
    try:
        async with temporary_audio_file(file) as tmp_path:
            file_size_bytes = os.path.getsize(tmp_path)
            file_size_mb = file_size_bytes / (1024 * 1024)
            if file_size_mb > config.MAX_FILE_SIZE_MB:
                raise HTTPException(
                    status_code=413,
                    detail=f"File is too large ({file_size_mb:.2f}MB). Maximum size is {config.MAX_FILE_SIZE_MB}MB.",
                )

            if not _is_audio_by_ffprobe(tmp_path):
                raise HTTPException(status_code=415, detail="Uploaded file does not appear to be an audio file.")

            estimated_memory = file_size_mb * 4  # conservative
            railway_memory_limit = int(os.getenv("RAILWAY_RAM_MB", "8192"))
            if estimated_memory > railway_memory_limit * 0.8:
                logger.warning(f"File {file.filename} ~{estimated_memory:.2f}MB est. memory (close to limit)")
            if not check_memory_availability(estimated_memory):
                metrics.record_error(endpoint_name, "insufficient_memory")
                raise HTTPException(
                    status_code=507,
                    detail=f"Insufficient memory to process this file ({file_size_mb:.2f}MB). Try a smaller file or try again later.",
                )

            if config.cache_enabled:
                fhash = get_file_hash(tmp_path)
                cached = request.app.state.cache.get(fhash)
                if cached:
                    logger.info(f"Returning cached result for '{file.filename}'")
                    metrics.record_processing_time(endpoint_name, 0.0)
                    return cached

            loop = asyncio.get_running_loop()
            executor = request.app.state.executor
            try:
                processing_result = await asyncio.wait_for(
                    loop.run_in_executor(executor, lambda: process_audio_file_sync(tmp_path, debug=debug)),
                    timeout=config.PROCESSING_TIMEOUT,
                )
            except asyncio.TimeoutError:
                metrics.record_error(endpoint_name, "timeout")
                raise ProcessingTimeoutError(f"Timed out after {config.PROCESSING_TIMEOUT}s")

            processing_time = time.time() - start_time
            metrics.record_processing_time(endpoint_name, processing_time)

            tempo_data = processing_result.get("tempo")
            tempo_out = tempo_data if isinstance(tempo_data, dict) else ({"bpm": tempo_data} if tempo_data else None)

            response = {
                "metadata": {
                    "filename": file.filename,
                    "processingTime": round(processing_time, 2),
                    **processing_result.get("metadata", {}),
                },
                "chords": processing_result.get("chords", []),
                "melody": processing_result.get("melody", []),
                "tempo": tempo_out,
            }
            if debug and "debug" in processing_result:
                response["debug"] = processing_result["debug"]

            if config.cache_enabled:
                try:
                    request.app.state.cache.set(fhash, response)
                except Exception:
                    pass
            return response

    except HTTPException:
        raise
    except AudioProcessingError:
        metrics.record_error(endpoint_name, "processing_error")
        raise
    except Exception as e:
        logger.error(f"Unexpected error in transcribe_audio: {e}", exc_info=True)
        metrics.record_error(endpoint_name, "unexpected_error")
        raise AudioProcessingError(f"Transcription failed: {e}") from e

@app.options("/transcribe", summary="CORS Preflight for Transcribe", tags=["Transcription"])
async def transcribe_options():
    """Handle CORS preflight requests for the transcribe endpoint."""
    return {"message": "OK"}

@app.options("/{full_path:path}", summary="Handle CORS Preflight", include_in_schema=False)
async def options_handler(full_path: str):
    """Handle CORS preflight requests for all endpoints."""
    return {"message": "OK"}

@app.post("/test-upload", 
    summary="Test File Upload Without Processing", 
    tags=["Diagnostics"],
    include_in_schema=config.ENABLE_DEBUG_ENDPOINTS
)
@limiter.limit("5/minute")
async def test_upload(request: Request, file: UploadFile = Depends(validate_file)):
    """
    Test endpoint that accepts a file upload but doesn't process it.
    Used to test if the issue is with file upload or audio processing.
    """
    require_debug_enabled()
    
    start_time = time.time()
    tmp_path = None

    try:
        # Save uploaded file to a temporary location using chunked streaming for memory efficiency
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(file.filename).suffix) as tmp_file:
            tmp_path = tmp_file.name
        
        # Use chunked streaming to avoid loading large files into memory
        await file.seek(0)  # Ensure reading from the start
        async with aiofiles.open(tmp_path, 'wb') as out_file:
            while content := await file.read(1024 * 1024):  # Read in 1MB chunks
                await out_file.write(content)
        
        # Get file info
        file_size = os.path.getsize(tmp_path)
        processing_time = time.time() - start_time
        
        logger.info(f"Successfully uploaded '{file.filename}' ({file_size / (1024*1024):.2f} MB) in {processing_time:.2f}s")
        
        return {
            "success": True,
            "message": "File uploaded successfully (no processing)",
            "filename": file.filename,
            "size_mb": round(file_size / (1024*1024), 2),
            "upload_time": round(processing_time, 2)
        }

    finally:
        # Ensure the temporary file is always cleaned up
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
                logger.debug(f"Cleaned up temporary file: {tmp_path}")
            except OSError as e:
                logger.warning(f"Failed to clean up temporary file {tmp_path}: {e}")

@app.get("/debug", summary="Debug Information", tags=["Status"])
async def debug_info():
    """Returns detailed debug information about the server state."""
    require_debug_enabled()
    import sys
    import platform
    
    debug_data = {
        "python_version": sys.version,
        "platform": platform.platform(),
        "dependencies_loaded": DEPENDENCIES_LOADED,
        "models_loaded": MODELS_LOADED,
        "config": {
            "max_file_size_mb": config.MAX_FILE_SIZE_MB,
            "max_workers": config.MAX_WORKERS,
            "processing_timeout": config.PROCESSING_TIMEOUT,
            "environment": config.ENVIRONMENT,
        },
        "supported_formats": list(config.ALLOWED_EXTENSIONS),
    }
    
    # Try to get ML library versions if available
    if DEPENDENCIES_LOADED:
        try:
            import librosa
            import numpy as np
            debug_data["ml_versions"] = {
                "librosa": librosa.__version__,
                "numpy": np.__version__,
            }
        except:
            debug_data["ml_versions"] = "Error getting versions"
    
    return debug_data

@app.get("/supported-formats", summary="Get Supported Formats", tags=["Status"])
async def get_supported_formats():
    """Returns the list of supported audio formats and file size limits."""
    formats = list(config.ALLOWED_EXTENSIONS)
    logger.info(f"Returning supported formats: {formats}")
    return {
        "supportedFormats": formats,
        "maxFileSizeMb": config.MAX_FILE_SIZE_MB
    }

@app.get("/metrics", summary="Get API Metrics", tags=["Status"])
async def get_metrics():
    """Return basic API usage metrics"""
    try:
        return app.state.metrics.get_stats()
    except Exception as e:
        logger.error(f"Error retrieving metrics: {e}")
        return {"error": "Failed to retrieve metrics"}

@app.get("/rate-limits", summary="Get Rate Limit Information", tags=["Status"])
async def get_rate_limits():
    """Return information about API rate limits"""
    return {
        "endpoints": {
            "/transcribe": "5 requests per minute",
            "/test-minimal-processing": "3 requests per minute", 
            "/test-dependencies": "5 requests per minute",
            "/transcribe-status": "10 requests per minute",
            "/test-upload": "5 requests per minute"
        },
        "rate_limit_headers": {
            "remaining": "X-RateLimit-Remaining",
            "reset": "X-RateLimit-Reset"
        },
        "note": "Rate limits are per IP address"
    }

# --- Debug Router Inclusion ---
if IS_DEV:
    try:
        from debug_routes import router as debug_router  # optional module
        app.include_router(debug_router, prefix="/_debug")
    except Exception:
        pass

# --- Main Execution Block ---
if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    # In a real production setup, you would use a process manager like Gunicorn or Uvicorn's --workers flag.
    # Map numeric logging level to the string uvicorn expects (e.g., "info").
    _level_name = logging.getLevelName(logging.getLogger().level)
    uvicorn.run(app, host="0.0.0.0", port=port, log_level=str(_level_name).lower())