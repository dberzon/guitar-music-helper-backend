from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import tempfile
import os
from typing import Optional

app = FastAPI(title="Guitar Music Helper Backend - Simple")

# Enable CORS for development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def read_root():
    return {"message": "Guitar Music Helper Backend is running!"}

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "dependencies_loaded": True,
        "models_loaded": True,
        "supported_formats": [".flac", ".wav", ".m4a", ".mp3", ".ogg"],
        "max_file_size_mb": 50
    }

@app.post("/transcribe")
async def transcribe_audio(
    source: str = Form(...),
    file: Optional[UploadFile] = File(None),
    url: Optional[str] = Form(None),
    youtube: Optional[str] = Form(None)
):
    """
    Simplified transcription endpoint that returns mock data for development.
    In production, this would use basic-pitch to analyze the audio.
    """
    
    if source == "file" and file:
        # For now, return mock chord progression data
        return {
            "chords": [
                {"time": 0.0, "duration": 2.0, "chord": "C"},
                {"time": 2.0, "duration": 2.0, "chord": "Am"},
                {"time": 4.0, "duration": 2.0, "chord": "F"},
                {"time": 6.0, "duration": 2.0, "chord": "G"},
                {"time": 8.0, "duration": 2.0, "chord": "C"},
            ],
            "tempo": {"bpm": 120, "confidence": 0.9},
            "metadata": {"duration": 10.0}
        }
    elif source == "url" and url:
        return {
            "chords": [
                {"time": 0.0, "duration": 1.5, "chord": "Em"},
                {"time": 1.5, "duration": 1.5, "chord": "C"},
                {"time": 3.0, "duration": 1.5, "chord": "G"},
                {"time": 4.5, "duration": 1.5, "chord": "D"},
            ],
            "tempo": {"bpm": 140, "confidence": 0.8},
            "metadata": {"duration": 6.0}
        }
    elif source == "youtube" and youtube:
        return {
            "chords": [
                {"time": 0.0, "duration": 4.0, "chord": "Dm"},
                {"time": 4.0, "duration": 4.0, "chord": "Bb"},
                {"time": 8.0, "duration": 4.0, "chord": "F"},
                {"time": 12.0, "duration": 4.0, "chord": "C"},
            ],
            "tempo": {"bpm": 110, "confidence": 0.7},
            "metadata": {"duration": 16.0}
        }
    else:
        raise HTTPException(status_code=400, detail="Invalid input parameters")

@app.post("/analyze/harmony")
async def analyze_harmony(request: dict):
    """
    Mock harmonic analysis endpoint that returns sample roman numeral analysis.
    """
    chords = request.get("chords", [])
    
    if not chords:
        return {
            "key": None,
            "roman": [],
            "modulations": []
        }
    
    # Mock analysis - in real implementation this would analyze the chord progression
    return {
        "key": {"name": "C", "mode": "major", "confidence": 0.9},
        "roman": [
            {"time": 0.0, "numeral": "I", "func": "T"},
            {"time": 2.0, "numeral": "vi", "func": "T"},
            {"time": 4.0, "numeral": "IV", "func": "S"},
            {"time": 6.0, "numeral": "V", "func": "D"},
            {"time": 8.0, "numeral": "I", "func": "T"},
        ],
        "modulations": []
    }

if __name__ == "__main__":
    uvicorn.run("simple_server:app", host="0.0.0.0", port=8000, reload=True)
