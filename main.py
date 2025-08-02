import os
import tempfile
import time
from typing import List, Optional
from pathlib import Path

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
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

# Configure CORS
cors_origins = [
    "http://localhost:3000", 
    "http://localhost:5173", 
    "http://localhost:5174"
]

# Add production origins if in production
if os.environ.get("RAILWAY_ENVIRONMENT") == "production":
    cors_origins.extend([
        "https://*.vercel.app",
        "https://*.netlify.app",
        "*"  # Allow all origins in production for now
    ])

app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
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
            os.unlink(tmp_path)
            
    except Exception as e:
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