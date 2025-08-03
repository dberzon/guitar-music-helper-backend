import os
import tempfile
import time
import asyncio
import logging
from typing import Dict
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, UploadFile, HTTPException, Request, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
import aiofiles
from pydantic_settings import BaseSettings
from pydantic import Field
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

# --- Configure Logging ---
# Initialize config first to get LOG_LEVEL
class Config(BaseSettings):
    MAX_FILE_SIZE_MB: int = Field(50, description="Max file size in MB")
    ALLOWED_EXTENSIONS: set[str] = {'.wav', '.mp3', '.m4a', '.flac', '.ogg'}
    MAX_WORKERS: int = 2
    ENVIRONMENT: str = Field("development", env="RAILWAY_ENVIRONMENT")
    LOG_LEVEL: str = "INFO"

    @property
    def MAX_FILE_SIZE(self):
        return self.MAX_FILE_SIZE_MB * 1024 * 1024

    class Config:
        env_file = ".env"

config = Config()

# Initialize rate limiter - limits requests based on client IP address
limiter = Limiter(key_func=get_remote_address)

# Configure logging using the config LOG_LEVEL
logging.basicConfig(level=config.LOG_LEVEL.upper())
logger = logging.getLogger(__name__)

# --- Application Configuration ---

# --- Lifespan Context Manager ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifespan - startup and shutdown events."""
    # Startup
    app.state.executor = ThreadPoolExecutor(max_workers=config.MAX_WORKERS)
    logger.info(f"ThreadPoolExecutor started with {config.MAX_WORKERS} workers")
    yield
    # Shutdown
    app.state.executor.shutdown(wait=True)
    logger.info("ThreadPoolExecutor shutdown complete")

# --- Thread Pool for CPU-intensive Tasks ---
# Note: executor is now managed via lifespan context and accessed via app.state.executor

# --- Custom Exceptions ---
class AudioProcessingError(Exception):
    """Custom exception for errors during the audio processing pipeline."""
    pass

class DependencyError(Exception):
    """Custom exception for when a required dependency is not available."""
    pass
# --- Dependency Loading with Graceful Failure ---
# Try to import heavy ML dependencies. If they fail, the app can still start,
# but the transcription endpoint will be disabled via dependency checks.
try:
    import librosa
    import numpy as np
    from basic_pitch import ICASSP_2022_MODEL_PATH
    from basic_pitch.inference import predict
    DEPENDENCIES_LOADED = True
    logger.info("All ML dependencies loaded successfully.")
except ImportError as e:
    logger.warning(f"Could not load ML dependencies. Transcription will be unavailable. Error: {e}")
    DEPENDENCIES_LOADED = False
    # Define dummy values to prevent runtime errors on startup
    librosa, np, predict = None, None, None

# Try to import local models and utilities.
try:
    # These are assumed to be Pydantic models in a local `models.py` file.
    from models import TranscriptionResponse, TranscriptionResult, TranscriptionMetadata
    # This is assumed to be a processing function in `transcription_utils.py`.
    from transcription_utils import process_basic_pitch_output
    MODELS_LOADED = True
    logger.info("Local models and utilities loaded successfully.")
except ImportError as e:
    logger.warning(f"Could not load local models/utils. Using dummy structures. Error: {e}")
    MODELS_LOADED = False
    # Define dummy structures for type hints and graceful failure
    TranscriptionResponse, TranscriptionResult, TranscriptionMetadata = dict, dict, dict
    process_basic_pitch_output = None

# --- FastAPI Application Setup ---
app = FastAPI(
    title="Guitar Music Helper - Audio Transcription API",
    description="An API to transcribe guitar audio into notes and chords using basic-pitch.",
    version="1.0.0",
    docs_url="/docs" if config.ENVIRONMENT == "development" else None,
    redoc_url="/redoc" if config.ENVIRONMENT == "development" else None,
    lifespan=lifespan,
)

# Configure rate limiting
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# --- CORS Middleware Configuration ---
# Always include production Vercel URLs to ensure they work regardless of environment detection
base_allowed_origins = [
    "https://guitar-music-helper.vercel.app",
    "https://guitar-music-helper-hq7lavby6-dberzons-projects.vercel.app",
    "https://guitar-music-helper-4b20ml8tu-dberzons-projects.vercel.app",
]

if config.ENVIRONMENT == "production":
    allowed_origins = base_allowed_origins
    allow_origin_regex = r"https://guitar-music-helper-.*\.vercel\.app$"
    allow_credentials = True
    logger.info(f"CORS configured for PRODUCTION.")
else:
    # More permissive CORS for development - allow all origins
    allowed_origins = ["*"]  # Allow all origins for development
    allow_origin_regex = None
    allow_credentials = False  # Must be False when allow_origins=["*"]
    logger.info(f"CORS configured for DEVELOPMENT with permissive settings.")

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_origin_regex=allow_origin_regex,
    allow_credentials=allow_credentials,
    allow_methods=["GET", "POST", "OPTIONS", "PUT", "DELETE"],
    allow_headers=["*"],
)

# --- Exception Handlers ---
# These handlers ensure that clients always receive a consistent,
# structured JSON error response.

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Handles any exception not caught by more specific handlers."""
    logger.error(f"Unhandled exception for request {request.url}: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "success": False,
            "error": {
                "code": "INTERNAL_SERVER_ERROR",
                "message": "An unexpected internal server error occurred.",
            },
        },
    )

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Handles FastAPI's built-in HTTPExceptions."""
    return JSONResponse(
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

@app.exception_handler(AudioProcessingError)
async def audio_processing_error_handler(request: Request, exc: AudioProcessingError):
    """Handles errors specific to the audio processing pipeline."""
    logger.warning(f"Audio processing error for request {request.url}: {exc}")
    return JSONResponse(
        status_code=422,  # Unprocessable Entity
        content={
            "success": False,
            "error": {
                "code": "AUDIO_PROCESSING_ERROR",
                "message": "Failed to process the provided audio file.",
                "details": str(exc),
            },
        },
    )

@app.exception_handler(DependencyError)
async def dependency_error_handler(request: Request, exc: DependencyError):
    """Handles errors when required libraries are not loaded."""
    logger.error(f"Dependency error for request {request.url}: {exc}")
    return JSONResponse(
        status_code=503,  # Service Unavailable
        content={
            "success": False,
            "error": {
                "code": "SERVICE_UNAVAILABLE",
                "message": "The transcription service is temporarily unavailable.",
                "details": str(exc),
            },
        },
    )
# --- Dependency Injection & Validation Helpers ---

def check_dependencies():
    """Dependency to ensure all required libraries are loaded before processing a request."""
    if not DEPENDENCIES_LOADED or not MODELS_LOADED:
        raise DependencyError("Required ML dependencies or local models are not loaded.")

def validate_file(file: UploadFile = File(...)) -> UploadFile:
    """
    Validates the uploaded file's existence, size, and extension.
    This runs as a dependency, cleaning up the endpoint logic.
    """
    if not file or not file.filename:
        raise HTTPException(status_code=400, detail="No file provided.")

    if file.size is not None and file.size > config.MAX_FILE_SIZE:
        size_mb = file.size / (1024 * 1024)
        raise HTTPException(
            status_code=413, # Payload Too Large
            detail=f"File is too large ({size_mb:.2f}MB). Maximum size is {config.MAX_FILE_SIZE_MB}MB."
        )

    if file.size == 0:
        raise HTTPException(
            status_code=400,
            detail="Empty file provided."
        )

    file_extension = Path(file.filename).suffix.lower()
    if file_extension not in config.ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=415,  # Unsupported Media Type
            detail=f"Unsupported file type '{file_extension}'. Allowed types are: {', '.join(config.ALLOWED_EXTENSIONS)}"
        )
    
    # Validate MIME type matches the file extension
    allowed_mime_types = {
        '.wav': 'audio/wav',
        '.mp3': 'audio/mpeg',
        '.m4a': 'audio/x-m4a',
        '.flac': 'audio/flac',
        '.ogg': 'audio/ogg'
    }
    expected = allowed_mime_types.get(file_extension)
    if expected and file.content_type != expected:
        raise HTTPException(
            status_code=415, 
            detail=f"MIME type {file.content_type} doesn't match expected {expected} for {file_extension} files"
        )
    
    return file

@app.get("/", summary="API Root", tags=["Status"])
async def root():
    """Provides basic service information and status."""
    return {
        "service": "Guitar Music Helper Audio Transcription API",
        "version": "1.0.0",
        "status": "healthy" if DEPENDENCIES_LOADED and MODELS_LOADED else "degraded",
    }

@app.get("/health", summary="Health Check", tags=["Status"])
async def health_check():
    """Performs a detailed health check of the service and its dependencies."""
    return {
        "status": "healthy" if DEPENDENCIES_LOADED and MODELS_LOADED else "degraded",
        "dependencies_loaded": DEPENDENCIES_LOADED,
        "models_loaded": MODELS_LOADED,
        "supported_formats": list(config.ALLOWED_EXTENSIONS),
        "max_file_size_mb": config.MAX_FILE_SIZE_MB,
        "timestamp": time.time(),
    }

# --- Synchronous Processing Function ---

def process_audio_file_sync(tmp_path: str) -> Dict:
    """
    Synchronous function to run in a thread pool. It loads an audio file,
    runs basic-pitch prediction, and processes the results.
    """
    try:
        logger.info(f"Starting audio processing at path: {tmp_path}")
        audio, sr = librosa.load(tmp_path, sr=None, mono=True)
        duration = librosa.get_duration(y=audio, sr=sr)
        logger.info(f"Audio loaded: duration={duration:.2f}s, sample_rate={sr}Hz")

        model_output, midi_data, note_events = predict(tmp_path)
        logger.info(f"Basic-pitch prediction complete. Found {len(note_events)} note events.")
        
        transcription_data = process_basic_pitch_output(
            model_output, midi_data, note_events, sr, duration
        )
        
        # Return a dictionary with the core results
        return {
            "metadata": {"duration": duration, "sampleRate": sr},
            "chords": transcription_data.get("chords", []),
            "melody": transcription_data.get("melody", []),
            "tempo": transcription_data.get("tempo"),
        }
    except Exception as e:
        logger.error(f"Core audio processing failed: {e}", exc_info=True)
        # Wrap the original exception in our custom error type
        raise AudioProcessingError(f"Prediction failed: {e}") from e

@app.post(
    "/transcribe",
    summary="Transcribe Audio File",
    tags=["Transcription"],
    dependencies=[Depends(check_dependencies)], # Protects the endpoint if dependencies are missing
)
@limiter.limit("5/minute")
async def transcribe_audio(request: Request, file: UploadFile = Depends(validate_file)):
    """
    Accepts an audio file, transcribes it to find chords and melody,
    and returns the structured data.
    """
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
        
        logger.info(f"File '{file.filename}' saved to temp path: {tmp_path} using chunked streaming")
        
        # Run the blocking, CPU-intensive function in the thread pool
        executor = request.app.state.executor
        loop = asyncio.get_event_loop()
        processing_result_dict = await loop.run_in_executor(
            executor, process_audio_file_sync, tmp_path
        )
        
        processing_time = time.time() - start_time
        logger.info(f"Successfully processed '{file.filename}' in {processing_time:.2f}s")

        # Construct the final Pydantic response model
        metadata = TranscriptionMetadata(
            filename=file.filename,
            processingTime=processing_time,
            **processing_result_dict["metadata"],
        )
        
        # Handle tempo which might be None
        tempo_data = processing_result_dict.get("tempo")
        result = TranscriptionResult(
            metadata=metadata,
            chords=processing_result_dict["chords"],
            melody=processing_result_dict["melody"],
            tempo=tempo_data
        )
        return TranscriptionResponse(success=True, data=result, processingTime=processing_time)

    finally:
        # Ensure the temporary file is always cleaned up
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
                logger.debug(f"Cleaned up temporary file: {tmp_path}")
            except OSError as e:
                logger.warning(f"Failed to clean up temporary file {tmp_path}: {e}")

@app.options("/transcribe", summary="CORS Preflight for Transcribe", tags=["Transcription"])
async def transcribe_options():
    """Handle CORS preflight requests for the transcribe endpoint."""
    return {"message": "OK"}

@app.options("/{full_path:path}", summary="Handle CORS Preflight", include_in_schema=False)
async def options_handler(full_path: str):
    """Handle CORS preflight requests for all endpoints."""
    return {"message": "OK"}

@app.get("/supported-formats", summary="Get Supported Formats", tags=["Status"])
async def get_supported_formats():
    """Returns the list of supported audio formats and file size limits."""
    formats = list(config.ALLOWED_EXTENSIONS)
    logger.info(f"Returning supported formats: {formats}")
    return {
        "supportedFormats": formats,
        "maxFileSizeMb": config.MAX_FILE_SIZE_MB
    }


# --- Main Execution Block ---
if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    # In a real production setup, you would use a process manager like Gunicorn or Uvicorn's --workers flag.
    uvicorn.run(app, host="0.0.0.0", port=port, log_level=logging.getLogger().level)