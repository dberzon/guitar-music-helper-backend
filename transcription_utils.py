import numpy as np
from typing import Dict, List, Tuple, Any
import librosa

from models import TranscriptionNote, TranscriptionChord, TranscriptionTempo


def midi_to_frequency(midi_note: int) -> float:
    """Convert MIDI note number to frequency in Hz."""
    return 440.0 * (2.0 ** ((midi_note - 69) / 12.0))


def frequency_to_midi(frequency: float) -> int:
    """Convert frequency in Hz to MIDI note number."""
    return int(round(12 * np.log2(frequency / 440.0) + 69))


def estimate_tempo(audio: np.ndarray, sr: int) -> Tuple[float, float]:
    """Estimate tempo using librosa."""
    try:
        tempo, _ = librosa.beat.tempo(y=audio, sr=sr)
        # Return average tempo and high confidence
        return float(np.mean(tempo)), 0.8
    except Exception:
        return 120.0, 0.0  # Default fallback


def extract_chords_from_notes(notes: List[Dict[str, Any]], 
                            time_step: float = 0.5) -> List[Dict[str, Any]]:
    """Extract chord progressions from note data."""
    if not notes:
        return []
    
    # Group notes into time windows
    max_time = max(note['end_time'] for note in notes)
    chords = []
    
    for start_time in np.arange(0, max_time, time_step):
        end_time = start_time + time_step
        
        # Find notes active in this window
        active_notes = [
            note for note in notes 
            if note['start_time'] < end_time and note['end_time'] > start_time
        ]
        
        if active_notes:
            # Get unique pitches
            pitches = sorted(set(note['pitch'] for note in active_notes))
            
            # Simple chord detection - just use pitch class names
            pitch_names = [librosa.midi_to_note(pitch, octave=False) for pitch in pitches]
            
            # Create basic chord symbol (this is simplified)
            if len(pitches) >= 3:
                chord_name = f"{pitch_names[0]}maj"  # Simplified major chord
            elif len(pitches) == 2:
                chord_name = f"{pitch_names[0]}{pitch_names[1]}"  # Interval
            else:
                chord_name = pitch_names[0]  # Single note
            
            chords.append({
                'time': start_time,
                'duration': time_step,
                'chord': chord_name,
                'confidence': np.mean([note['confidence'] for note in active_notes])
            })
    
    return chords


def process_basic_pitch_output(model_output: Dict[str, Any], 
                             midi_data: Any, 
                             note_events: List[Tuple],
                             sample_rate: int,
                             duration: float) -> Dict[str, Any]:
    """Process basic-pitch output into our transcription format."""
    
    # Extract note events
    notes = []
    for note in note_events:
        if len(note) >= 4:
            start_time, end_time, pitch, amplitude = note[:4]
            notes.append({
                'start_time': start_time,
                'end_time': end_time,
                'pitch': int(pitch),
                'amplitude': float(amplitude),
                'confidence': float(amplitude)  # Use amplitude as confidence
            })
    
    # Convert to our format
    transcription_notes = [
        TranscriptionNote(
            time=note['start_time'],
            duration=note['end_time'] - note['start_time'],
            pitch=note['pitch'],
            frequency=midi_to_frequency(note['pitch']),
            confidence=note['confidence']
        )
        for note in notes
    ]
    
    # Extract chords
    chord_data = extract_chords_from_notes(notes)
    transcription_chords = [
        TranscriptionChord(
            time=chord['time'],
            duration=chord['duration'],
            chord=chord['chord'],
            confidence=chord['confidence']
        )
        for chord in chord_data
    ]
    
    # Estimate tempo
    tempo = None
    # Note: In a real implementation, we'd load the audio here to estimate tempo
    # For now, we'll skip tempo estimation to avoid loading audio twice
    
    return {
        "melody": transcription_notes,
        "chords": transcription_chords,
        "tempo": tempo
    }


def convert_to_transcription_format(basic_pitch_output: Dict[str, Any], 
                                  audio_path: str) -> Dict[str, Any]:
    """Convert basic-pitch output to our transcription format.
    Loads audio at sr=22050 mono to align with the processing pipeline.
    """
    audio, sr = librosa.load(audio_path, sr=22050, mono=True)
    if audio.size == 0:
        tempo_bpm, tempo_conf = 120.0, 0.0
    else:
        audio = np.asarray(audio, dtype=np.float32, order="C")
        tempo_bpm, tempo_conf = estimate_tempo(audio, sr)
    return {
        "melody": [],
        "chords": [],
        "tempo": {"bpm": float(tempo_bpm), "confidence": float(tempo_conf)},
    }


def filter_notes_by_confidence(notes: List[TranscriptionNote], 
                             min_confidence: float = 0.5) -> List[TranscriptionNote]:
    """Filter notes by confidence threshold."""
    return [note for note in notes if note.confidence >= min_confidence]


def group_notes_by_time(notes: List[TranscriptionNote], 
                       time_window: float = 0.1) -> List[List[TranscriptionNote]]:
    """Group notes into time windows for chord analysis."""
    if not notes:
        return []
    
    max_time = max(note.time + note.duration for note in notes)
    grouped = []
    
    current_time = 0
    while current_time < max_time:
        window_notes = [
            note for note in notes
            if note.time <= current_time + time_window and 
               note.time + note.duration >= current_time
        ]
        if window_notes:
            grouped.append(window_notes)
        current_time += time_window
    
    return grouped