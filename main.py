import os
import tempfile
import time
from typing import List, Optional
from pathlib import Path

from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
import uvicorn
import librosa
import numpy as np
from basic_pitch import ICASSP_2022_MODEL_PATH
from basic_pitch.inference import predict
import scipy.io.wavfile as wavfile

from models import TranscriptionResponse, TranscriptionResult, TranscriptionMetadata
from transcription_utils import process_basic_pitch_output, convert_to_transcription_format

app = FastAPI(
    title="Guitar Music Helper - Audio Transcription API",
    description="Audio transcription service for guitar music using basic-pitch",
    version="1.0.0"
)

# Configure CORS - Temporarily more permissive for debugging
allowed_origins = [
    "*",  # Temporarily allow all origins
]

# Allow all Vercel and Netlify subdomains
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=False,  # Must be False when using "*"
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
    
    print(f"Request: {method} {url}")
    print(f"Origin: {origin}")
    print(f"Headers: {dict(request.headers)}")
    
    response = await call_next(request)
    
    print(f"Response status: {response.status_code}")
    print(f"Response headers: {dict(response.headers)}")
    
    return response

# Global exception handler to ensure CORS headers are always present
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Handle all exceptions and ensure CORS headers are present."""
    print(f"Global exception: {str(exc)}")
    print(f"Exception type: {type(exc).__name__}")
    
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
    
    # Add CORS headers manually for wildcard
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, PUT, DELETE, OPTIONS"
    response.headers["Access-Control-Allow-Headers"] = "*"
    
    return response

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Handle HTTP exceptions and ensure CORS headers are present."""
    response = JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail}
    )
    
    # Add CORS headers manually for wildcard
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, PUT, DELETE, OPTIONS"
    response.headers["Access-Control-Allow-Headers"] = "*"
    
    return response
# Initialize basic-pitch model
model_path = ICASSP_2022_MODEL_PATH

# Constants
MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB
ALLOWED_EXTENSIONS = {'.wav', '.mp3', '.m4a', '.flac', '.ogg'}

@app.get("/")
async def root():
    return {"Hello": "World", "service": "Guitar Music Helper Audio Transcription API", "version": "1.0.0"}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "model_loaded": True}

@app.get("/cors-test")
async def cors_test():
    """Simple endpoint to test CORS configuration."""
    return {"message": "CORS is working", "timestamp": time.time()}

@app.post("/transcribe", response_model=TranscriptionResponse)
async def transcribe_audio(file: UploadFile = File(...)):
    """Transcribe audio file to chords and melody."""
    start_time = time.time()
    
    try:
        # Validate file
        if not file.filename:
            raise HTTPException(status_code=400, detail="No file provided")
        
        file_extension = Path(file.filename).suffix.lower()
        if file_extension not in ALLOWED_EXTENSIONS:
            raise HTTPException(
                status_code=400, 
                detail=f"Unsupported file type. Allowed: {', '.join(ALLOWED_EXTENSIONS)}"
            )
        
        # Read file content
        content = await file.read()
        if len(content) > MAX_FILE_SIZE:
            raise HTTPException(status_code=400, detail="File too large. Max 50MB")
        
        # Create temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as tmp_file:
            tmp_file.write(content)
            tmp_path = tmp_file.name
        
        try:
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
                filename=file.filename,
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
            
        finally:
            # Clean up temporary file
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
            
    except HTTPException:
        # Re-raise HTTP exceptions as-is
        raise
    except Exception as e:
        # Log the error for debugging
        print(f"Transcription error: {str(e)}")
        print(f"Error type: {type(e).__name__}")
        
        return TranscriptionResponse(
            success=False,
            error={
                "code": "TRANSCRIPTION_ERROR",
                "message": str(e),
                "details": {"type": type(e).__name__}
            },
            processingTime=time.time() - start_time
        )

@app.get("/supported-formats")
async def get_supported_formats():
    """Get list of supported audio formats."""
    return {
        "formats": list(ALLOWED_EXTENSIONS),
        "max_file_size": MAX_FILE_SIZE
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", default=8000)), log_level="info")