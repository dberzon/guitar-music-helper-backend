from typing import List, Optional
from pydantic import BaseModel, Field

class TranscriptionNote(BaseModel):
    """A single transcribed note."""
    time: float = Field(..., description="Time in seconds from start of audio")
    duration: float = Field(..., description="Duration in seconds")
    pitch: int = Field(..., description="MIDI note number (0-127)", ge=0, le=127)
    frequency: float = Field(..., description="Frequency in Hz")
    confidence: float = Field(..., description="Confidence score (0-1)", ge=0, le=1)

class TranscriptionChord(BaseModel):
    """A detected chord."""
    time: float = Field(..., description="Time in seconds from start of audio")
    duration: float = Field(..., description="Duration in seconds")
    chord: str = Field(..., description="Chord symbol (e.g., 'Cmaj7', 'Am', 'G/B')")
    confidence: float = Field(..., description="Confidence score (0-1)", ge=0, le=1)

class TranscriptionTempo(BaseModel):
    """Detected tempo information."""
    bpm: float = Field(..., description="Beats per minute")
    confidence: float = Field(..., description="Confidence score (0-1)", ge=0, le=1)

class TranscriptionMetadata(BaseModel):
    """Metadata about the transcribed audio."""
    filename: str = Field(..., description="Original filename")
    duration: float = Field(..., description="Audio duration in seconds")
    sampleRate: int = Field(..., description="Sample rate in Hz")
    processingTime: float = Field(..., description="Processing time in seconds")

class TranscriptionResult(BaseModel):
    """Complete transcription result."""
    metadata: TranscriptionMetadata
    chords: List[TranscriptionChord]
    melody: List[TranscriptionNote]
    tempo: Optional[TranscriptionTempo] = None

class TranscriptionError(BaseModel):
    """Error information for failed transcription."""
    code: str = Field(..., description="Error code")
    message: str = Field(..., description="Human-readable error message")
    details: Optional[dict] = Field(None, description="Additional error details")

class TranscriptionResponse(BaseModel):
    """Response wrapper for transcription requests."""
    success: bool
    data: Optional[TranscriptionResult] = None
    error: Optional[TranscriptionError] = None
    processingTime: Optional[float] = Field(None, description="Processing time in seconds")