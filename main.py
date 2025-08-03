import os
import tempfile
import time
import uuid
import asyncio
import logging
from typing import List, Optional
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

from fastapi import FastAPI, File, UploadFile, HTTPException, Request, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
import uvicorn

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
class Config:
    MAX_FILE_SIZE_MB = int(os.getenv("MAX_FILE_SIZE_MB", "50"))
    MAX_FILE_SIZE = MAX_FILE_SIZE_MB * 1024 * 1024
    ALLOWED_EXTENSIONS = {'.wav', '.mp3', '.m4a', '.flac', '.ogg'}
    MAX_WORKERS = int(os.getenv("MAX_WORKERS", "2"))  # Conservative for Railway
    ENVIRONMENT = os.getenv("RAILWAY_ENVIRONMENT", "development")

config = Config()

# Thread pool for CPU-intensive tasks
executor = ThreadPoolExecutor(max_workers=config.MAX_WORKERS)

# Custom exceptions
class AudioProcessingError(Exception):
    pass

class DependencyError(Exception):
    pass
# Try to import heavy dependencies with error handling
try:
    import librosa
    import numpy as np
    from basic_pitch import ICASSP_2022_MODEL_PATH
    from basic_pitch.inference import predict
    import scipy.io.wavfile as wavfile
    DEPENDENCIES_LOADED = True
    logger.info("All ML dependencies loaded successfully")
except Exception as e:
    logger.warning(f"Could not load ML dependencies: {e}")
    DEPENDENCIES_LOADED = False
    # Set dummy values to prevent import errors
    ICASSP_2022_MODEL_PATH = None
    predict = None

# Import models and utilities
try:
    from models import TranscriptionResponse, TranscriptionResult, TranscriptionMetadata
    from transcription_utils import process_basic_pitch_output, convert_to_transcription_format
    MODELS_LOADED = True
    logger.info("Models and utilities loaded successfully")
except Exception as e:
    logger.warning(f"Could not load models: {e}")
    MODELS_LOADED = False
    # These will be needed for type hints even if not working
    TranscriptionResponse = dict
    TranscriptionResult = dict
    TranscriptionMetadata = dict
    process_basic_pitch_output = None

app = FastAPI(
    title="Guitar Music Helper - Audio Transcription API",
    description="Audio transcription service for guitar music using basic-pitch",
    version="1.0.0"
)

# Configure CORS based on environment
if config.ENVIRONMENT == "production":
    allowed_origins = [
        "https://guitar-music-helper.vercel.app",
        "https://guitar-music-helper-git-main.vercel.app",
    ]
    # Allow Vercel preview deployments
    allow_origin_regex = r"https://guitar-music-helper-.*\.vercel\.app$"
    allow_credentials = True
else:
    # Development - more permissive
    allowed_origins = [
        "http://localhost:3000",
        "http://localhost:5173", 
        "http://localhost:5174",
        "*"  # Temporarily for debugging
    ]
    allow_origin_regex = None
    allow_credentials = False

logger.info(f"CORS configured for {config.ENVIRONMENT} environment")

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_origin_regex=allow_origin_regex,
    allow_credentials=allow_credentials,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

# Debug middleware to log requests
@app.middleware("http")
async def debug_middleware(request: Request, call_next):
    """Debug middleware to log requests and origins."""
    origin = request.headers.get("origin", "No origin header")
    method = request.method
    url = str(request.url)
    
    logger.debug(f"Request: {method} {url}")
    logger.debug(f"Origin: {origin}")
    logger.debug(f"Headers: {dict(request.headers)}")
    
    response = await call_next(request)
    
    logger.debug(f"Response status: {response.status_code}")
    logger.debug(f"Response headers: {dict(response.headers)}")
    
    return response

# Global exception handler to ensure CORS headers are always present
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Handle all exceptions and ensure CORS headers are present."""
    logger.error(f"Global exception: {str(exc)}", exc_info=True)
    
    response = JSONResponse(
        status_code=500,
        content={
            "success": False,
            "error": {
                "code": "INTERNAL_SERVER_ERROR",
                "message": "An internal server error occurred",
                "details": {"type": type(exc).__name__}
            }
        }
    )
    
    return response

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Handle HTTP exceptions and ensure CORS headers are present."""
    logger.warning(f"HTTP exception: {exc.status_code} - {exc.detail}")
    
    response = JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail}
    )
    
    return response

@app.exception_handler(AudioProcessingError)
async def audio_processing_error_handler(request: Request, exc: AudioProcessingError):
    """Handle audio processing errors."""
    logger.warning(f"Audio processing error: {str(exc)}")
    
    response = JSONResponse(
        status_code=400,
        content={
            "success": False,
            "error": {
                "code": "AUDIO_PROCESSING_ERROR",
                "message": "Failed to process audio file",
                "details": {"type": type(exc).__name__}
            }
        }
    )
    
    return response
# Initialize basic-pitch model
model_path = ICASSP_2022_MODEL_PATH if DEPENDENCIES_LOADED else None

# Dependency injection helpers
def get_dependencies_status():
    """Check if all dependencies are available."""
    if not DEPENDENCIES_LOADED:
        raise HTTPException(
            status_code=503, 
            detail="ML dependencies not loaded - transcription service unavailable"
        )
    if not MODELS_LOADED:
        raise HTTPException(
            status_code=503, 
            detail="Model dependencies not loaded - transcription service unavailable"
        )
    return True

def validate_file_size(file_size: int):
    """Validate file size against limits."""
    if file_size > config.MAX_FILE_SIZE:
        raise HTTPException(
            status_code=413, 
            detail=f"File too large. Maximum size: {config.MAX_FILE_SIZE_MB}MB"
        )

def validate_file_extension(filename: str):
    """Validate file extension."""
    file_extension = Path(filename).suffix.lower()
    if file_extension not in config.ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400, 
            detail=f"Unsupported file type. Allowed: {', '.join(config.ALLOWED_EXTENSIONS)}"
        )
    return file_extension

@app.get("/")
async def root():
    return {
        "Hello": "World", 
        "service": "Guitar Music Helper Audio Transcription API", 
        "version": "1.0.0",
        "dependencies_loaded": DEPENDENCIES_LOADED
    }

@app.get("/health")
async def health_check():
    return {
        "status": "healthy", 
        "model_loaded": DEPENDENCIES_LOADED and MODELS_LOADED,
        "dependencies_status": "loaded" if DEPENDENCIES_LOADED else "failed_to_load",
        "models_status": "loaded" if MODELS_LOADED else "failed_to_load"
    }

@app.get("/cors-test")
async def cors_test():
    """Simple endpoint to test CORS configuration."""
    return {"message": "CORS is working", "timestamp": time.time()}

def process_audio_file_sync(tmp_path: str, filename: str, start_time: float):
    """Process audio file synchronously (to be run in thread)."""
    try:
        # Ensure all required functions are available
        if not librosa or not predict or not process_basic_pitch_output:
            raise AudioProcessingError("Required ML dependencies not properly loaded")
            
        # Load audio
        audio, sr = librosa.load(tmp_path, sr=None, mono=True)
        
        # Get audio duration
        duration = len(audio) / sr
        
        # Run basic-pitch prediction
        model_output, midi_data, note_events = predict(tmp_path)
        
        # Process the output
        transcription_data = process_basic_pitch_output(
            model_output, 
            midi_data, 
            note_events,
            sr,
            duration
        )
        
        # Create metadata
        metadata = TranscriptionMetadata(
            filename=filename,
            duration=duration,
            sampleRate=sr,
            processingTime=time.time() - start_time
        )
        
        # Create final result
        result = TranscriptionResult(
            metadata=metadata,
            chords=transcription_data["chords"],
            melody=transcription_data["melody"],
            tempo=transcription_data.get("tempo")
        )
        
        return TranscriptionResponse(
            success=True,
            data=result,
            processingTime=time.time() - start_time
        )
        
    except Exception as e:
        logger.error(f"Audio processing failed: {str(e)}", exc_info=True)
        raise AudioProcessingError(f"Failed to process audio: {str(e)}")

@app.post("/transcribe")
async def transcribe_audio(
    file: UploadFile = File(...),
    _deps_check: bool = Depends(get_dependencies_status)
):
    """Transcribe audio file to chords and melody."""
    start_time = time.time()
    tmp_path = None
    
    try:
        # Validate file
        if not file.filename:
            raise HTTPException(status_code=400, detail="No file provided")
        
        # Use validation functions
        file_extension = validate_file_extension(file.filename)
        
        # Stream file to disk instead of loading into memory
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as tmp_file:
            tmp_path = tmp_file.name
            content_size = 0
            chunk_size = 1024 * 1024  # 1MB chunks
            
            while True:
                chunk = await file.read(chunk_size)
                if not chunk:
                    break
                content_size += len(chunk)
                # Check size incrementally to avoid memory issues
                if content_size > config.MAX_FILE_SIZE:
                    raise HTTPException(
                        status_code=413, 
                        detail=f"File too large. Maximum size: {config.MAX_FILE_SIZE_MB}MB"
                    )
                tmp_file.write(chunk)
        
        # Process audio in a separate thread to avoid blocking the event loop
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            executor, 
            process_audio_file_sync,
            tmp_path, 
            file.filename, 
            start_time
        )
        
        return result
        
    except HTTPException:
        # Re-raise HTTP exceptions as-is
        raise
    except AudioProcessingError:
        # Re-raise audio processing errors as-is
        raise
    except Exception as e:
        # Log unexpected errors
        logger.error(f"Unexpected transcription error: {str(e)}", exc_info=True)
        
        # Return consistent error response format
        raise HTTPException(
            status_code=500,
            detail={
                "success": False,
                "error": {
                    "code": "TRANSCRIPTION_ERROR",
                    "message": str(e),
                    "details": {"type": type(e).__name__}
                },
                "processingTime": time.time() - start_time
            }
        )
    finally:
        # Clean up temporary file
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
            except Exception as e:
                logger.warning(f"Failed to clean up temporary file {tmp_path}: {e}")

@app.get("/supported-formats")
async def get_supported_formats():
    """Get list of supported audio formats."""
    return {
        "formats": list(config.ALLOWED_EXTENSIONS),
        "max_file_size": config.MAX_FILE_SIZE,
        "max_file_size_mb": config.MAX_FILE_SIZE_MB
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", default=8000)), log_level="info")