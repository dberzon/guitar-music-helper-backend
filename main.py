import os
import tempfile
import time
import asyncio
import logging
from typing import Dict, TYPE_CHECKING
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

# Type hints for static analysis even when imports fail
if TYPE_CHECKING:
    from models import TranscriptionResponse, TranscriptionResult, TranscriptionMetadata
    from transcription_utils import process_basic_pitch_output

# --- Constants ---
ALLOWED_MIME_TYPES = {
    '.wav': ['audio/wav', 'audio/x-wav'],
    '.mp3': ['audio/mpeg', 'audio/mp3', 'application/octet-stream'],  # Allow octet-stream for curl uploads
    '.m4a': ['audio/x-m4a', 'audio/m4a', 'audio/mp4', 'application/octet-stream'],
    '.flac': ['audio/flac', 'audio/x-flac', 'application/octet-stream'],
    '.ogg': ['audio/ogg', 'application/ogg', 'application/octet-stream']
}

# --- Configure Logging ---
# Initialize config first to get LOG_LEVEL
class Config(BaseSettings):
    MAX_FILE_SIZE_MB: int = Field(10, description="Max file size in MB - reduced for Railway memory limits")
    ALLOWED_EXTENSIONS: set[str] = {'.wav', '.mp3', '.m4a', '.flac', '.ogg'}
    MAX_WORKERS: int = 1  # Reduced to prevent memory exhaustion
    ENVIRONMENT: str = Field("development", env="RAILWAY_ENVIRONMENT")
    LOG_LEVEL: str = "INFO"
    PROCESSING_TIMEOUT: int = Field(45, description="Max processing time in seconds - increased for Railway")
    CORS_ORIGINS: list[str] = Field(
        default=["http://localhost:5173", "http://127.0.0.1:5173", "http://localhost:5174", "http://127.0.0.1:5174"],
        description="List of allowed CORS origins"
    )
    CORS_ORIGIN_REGEX: str | None = Field(
        default=r"^https://guitar-music-helper-[a-z0-9]+-dberzons-projects\.vercel\.app$",
        description="Regex for Vercel preview deployments"
    )

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
    # No need to define dummy structures - TYPE_CHECKING provides types for IDEs

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
# Configure CORS with environment-based origins for security
logger.info(f"🌐 CORS configured for {config.ENVIRONMENT} environment")

app.add_middleware(
    CORSMiddleware,
    allow_origins=config.CORS_ORIGINS,
    allow_origin_regex=config.CORS_ORIGIN_REGEX,
    allow_credentials=True,  # Can be True now that we don't use "*"
    allow_methods=["GET", "POST", "OPTIONS"],  # Be specific about allowed methods
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
    
    # Validate MIME type matches the file extension - allow common variations
    allowed_types = ALLOWED_MIME_TYPES.get(file_extension, [])
    if allowed_types and file.content_type not in allowed_types:
        raise HTTPException(
            status_code=415, 
            detail=f"MIME type {file.content_type} not allowed for {file_extension} files. Allowed: {', '.join(allowed_types)}"
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
    import gc  # Import garbage collector for memory management
    
    try:
        logger.info(f"Starting audio processing at path: {tmp_path}")
        
        # Check file size before processing
        file_size = os.path.getsize(tmp_path)
        logger.info(f"File size: {file_size / (1024*1024):.2f} MB")
        
        # Force garbage collection before starting
        gc.collect()
        
        # Load audio with memory monitoring - use lower sample rate to save memory
        logger.info("Loading audio with librosa...")
        audio, sr = librosa.load(tmp_path, sr=22050, mono=True)  # Reduced sample rate
        duration = librosa.get_duration(y=audio, sr=sr)
        logger.info(f"Audio loaded: duration={duration:.2f}s, sample_rate={sr}Hz, audio_shape={audio.shape}")

        # Run basic-pitch prediction
        logger.info("Starting basic-pitch prediction...")
        model_output, midi_data, note_events = predict(tmp_path)
        logger.info(f"Basic-pitch prediction complete. Found {len(note_events)} note events.")
        
        # Clean up audio data from memory
        del audio
        gc.collect()
        
        # Process the output
        logger.info("Processing basic-pitch output...")
        transcription_data = process_basic_pitch_output(
            model_output, midi_data, note_events, sr, duration
        )
        logger.info("Processing complete, returning results...")
        
        # Clean up intermediate data
        del model_output, midi_data, note_events
        gc.collect()
        
        # Return a dictionary with the core results
        return {
            "metadata": {"duration": duration, "sampleRate": sr},
            "chords": transcription_data.get("chords", []),
            "melody": transcription_data.get("melody", []),
            "tempo": transcription_data.get("tempo"),
        }
    except Exception as e:
        logger.error(f"Core audio processing failed: {e}", exc_info=True)
        # Force garbage collection on error
        gc.collect()
        # Wrap the original exception in our custom error type
        raise AudioProcessingError(f"Prediction failed: {e}") from e

@app.post("/test-minimal-processing", summary="Test Minimal Audio Processing", tags=["Testing"])
@limiter.limit("3/minute")
async def test_minimal_processing(request: Request, file: UploadFile = Depends(validate_file)):
    """
    Test minimal audio processing to identify where exactly the failure occurs.
    """
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
            
            audio, sr = librosa.load(tmp_path, sr=22050, mono=True, duration=10.0)  # Limit to 10 seconds
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

@app.post("/test-dependencies", summary="Test ML Dependencies", tags=["Testing"])
@limiter.limit("5/minute")
async def test_dependencies(request: Request):
    """
    Test if ML dependencies can be imported and used without file processing.
    """
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
        file_size_mb = file.size / (1024 * 1024) if file.size else 0
        
        # Estimate memory requirements (rough calculation)
        # Basic-pitch typically needs 3-5x the audio file size in memory
        estimated_memory_mb = file_size_mb * 4  # Conservative estimate
        
        # Railway Hobby plan has ~512MB available memory
        railway_memory_limit = 500  # Conservative estimate
        
        return {
            "success": True,
            "file_info": {
                "filename": file.filename,
                "size_mb": round(file_size_mb, 2),
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
        
        # Run the blocking, CPU-intensive function in the thread pool with timeout
        executor = request.app.state.executor
        loop = asyncio.get_event_loop()
        
        # Add timeout to prevent Railway from timing out
        try:
            processing_result_dict = await asyncio.wait_for(
                loop.run_in_executor(executor, process_audio_file_sync, tmp_path),
                timeout=config.PROCESSING_TIMEOUT
            )
        except asyncio.TimeoutError:
            logger.error(f"Audio processing timed out after {config.PROCESSING_TIMEOUT} seconds for file: {file.filename}")
            raise AudioProcessingError(f"Audio processing timed out after {config.PROCESSING_TIMEOUT} seconds. Try uploading a smaller file.")
        
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

@app.post("/test-upload", summary="Test File Upload Without Processing", tags=["Testing"])
@limiter.limit("5/minute")
async def test_upload(request: Request, file: UploadFile = Depends(validate_file)):
    """
    Test endpoint that accepts a file upload but doesn't process it.
    Used to test if the issue is with file upload or audio processing.
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


# --- Main Execution Block ---
if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    # In a real production setup, you would use a process manager like Gunicorn or Uvicorn's --workers flag.
    uvicorn.run(app, host="0.0.0.0", port=port, log_level=logging.getLogger().level)