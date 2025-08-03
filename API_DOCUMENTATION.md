# API Documentation - Guitar Music Helper

## Overview
This document provides detailed API documentation for the Guitar Music Helper Audio Transcription API. It includes endpoint specifications, request/response formats, error handling, and usage examples.

---

## Base URL
```
http://localhost:8000
```

## Authentication
This API does not require authentication for basic usage.

---

## Endpoints

### 1. Health Check
Check if the API service is running and healthy.

**Endpoint:** `GET /health`

**Description:** Returns the current health status of the service.

**Request:**
```
GET /health
```

**Response (200 OK):**
```json
{
  "status": "healthy",
  "timestamp": "2023-12-07T10:30:00Z"
}
```

**Response Fields:**
- `status` (string): Always "healthy" when service is running
- `timestamp` (string): ISO 8601 formatted timestamp of the health check

---

### 2. Transcribe Audio
Transcribe an audio file to extract musical information including melody, chords, and tempo.

**Endpoint:** `POST /transcribe`

**Description:** Upload an audio file and receive detailed musical transcription data including detected notes, chord progressions, and tempo information.

**Request:**
```
POST /transcribe
Content-Type: multipart/form-data
```

**Parameters:**
| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| file | file | Yes | Audio file (MP3, WAV, M4A, FLAC) |
| min_confidence | float | No | Minimum confidence threshold (0.0-1.0, default: 0.5) |

**File Requirements:**
- **Supported Formats:** MP3, WAV, M4A, FLAC
- **Maximum Size:** 50MB
- **Recommended Quality:** 44.1kHz, 16-bit
- **Maximum Duration:** 5 minutes (for optimal performance)

**Response (200 OK):**
```json
{
  "melody": [
    {
      "time": 0.5,
      "duration": 0.25,
      "pitch": 64,
      "frequency": 329.63,
      "confidence": 0.95
    },
    {
      "time": 0.75,
      "duration": 0.5,
      "pitch": 67,
      "frequency": 392.0,
      "confidence": 0.88
    }
  ],
  "chords": [
    {
      "time": 0.0,
      "duration": 2.0,
      "chord": "Cmaj",
      "confidence": 0.92
    },
    {
      "time": 2.0,
      "duration": 2.0,
      "chord": "Gmaj",
      "confidence": 0.85
    }
  ],
  "tempo": {
    "bpm": 120.5,
    "confidence": 0.94
  }
}
```

**Response Fields:**

#### Melody Array
Each melody note contains:
- `time` (float): Start time in seconds from the beginning of the audio
- `duration` (float): Duration of the note in seconds
- `pitch` (int): MIDI note number (0-127, where 60 = Middle C)
- `frequency` (float): Frequency in Hz
- `confidence` (float): Detection confidence score (0.0-1.0)

#### Chords Array
Each chord contains:
- `time` (float): Start time in seconds
- `duration` (float): Duration of the chord in seconds
- `chord` (string): Chord symbol (e.g., "Cmaj", "Am", "G7")
- `confidence` (float): Detection confidence score (0.0-1.0)

#### Tempo Object
- `bpm` (float): Beats per minute
- `confidence` (float): Detection confidence score (0.0-1.0)

---

## Error Handling

### 400 Bad Request
Returned when the request is malformed or contains invalid data.

**Example Response:**
```json
{
  "detail": "Invalid file format. Supported formats: mp3, wav, m4a, flac"
}
```

### 413 Payload Too Large
Returned when the uploaded file exceeds the size limit.

**Example Response:**
```json
{
  "detail": "File too large. Maximum size: 50MB"
}
```

### 422 Unprocessable Entity
Returned when the file cannot be processed (corrupted, unsupported codec, etc.).

**Example Response:**
```json
{
  "detail": "Cannot process audio file: Unsupported audio codec"
}
```

### 500 Internal Server Error
Returned when an unexpected server error occurs.

**Example Response:**
```json
{
  "detail": "Internal server error during audio processing"
}
```

---

## Usage Examples

### cURL Example
```bash
# Basic transcription
curl -X POST "http://localhost:8000/transcribe" \
  -F "file=@guitar_sample.mp3"

# With custom confidence threshold
curl -X POST "http://localhost:8000/transcribe" \
  -F "file=@guitar_sample.mp3" \
  -F "min_confidence=0.7"
```

### Python Example
```python
import requests

# Basic usage
url = "http://localhost:8000/transcribe"
files = {"file": open("guitar_sample.mp3", "rb")}
response = requests.post(url, files=files)
data = response.json()

# With parameters
params = {"min_confidence": 0.7}
response = requests.post(url, files=files, params=params)
data = response.json()

# Access results
melody_notes = data["melody"]
chords = data["chords"]
tempo = data["tempo"]["bpm"]
```

### JavaScript Example
```javascript
// Using fetch API
const formData = new FormData();
formData.append('file', audioFile);
formData.append('min_confidence', '0.7');

fetch('http://localhost:8000/transcribe', {
  method: 'POST',
  body: formData
})
.then(response => response.json())
.then(data => {
  console.log('Melody:', data.melody);
  console.log('Chords:', data.chords);
  console.log('Tempo:', data.tempo.bpm);
});
```

---

## Rate Limiting
Currently, there are no rate limits on the API. However, for production use, consider implementing:
- Request rate limiting per IP
- File size and upload frequency limits
- Queue system for large files

---

## Performance Guidelines

### Optimal File Characteristics
- **Duration**: 10-30 seconds for fastest processing
- **Format**: WAV files process faster than compressed formats
- **Quality**: 44.1kHz, 16-bit provides best balance of quality and processing speed
- **Content**: Clear, isolated guitar audio yields best results

### Expected Processing Times
| File Size | Duration | Processing Time |
|-----------|----------|-----------------|
| < 1MB | < 10s | 1-3 seconds |
| 1-5MB | 10-30s | 3-8 seconds |
| 5-20MB | 30s-2min | 8-20 seconds |
| 20-50MB | 2-5min | 20-60 seconds |

---

## Chord Detection Details

### Supported Chord Types
- Major chords: "Cmaj", "Gmaj", etc.
- Minor chords: "Am", "Em", etc.
- Dominant 7th: "G7", "C7", etc.
- Major 7th: "CM7", "FM7", etc.
- Minor 7th: "Am7", "Dm7", etc.

### Chord Detection Algorithm
1. **Time Windowing**: Audio is divided into 0.5-second windows
2. **Pitch Detection**: Identifies all active pitches in each window
3. **Chord Matching**: Matches pitch combinations to known chord patterns
4. **Confidence Scoring**: Assigns confidence based on pitch clarity and chord complexity

---

## MIDI Note Reference

### Common Guitar Notes
| String | Fret 0 | Fret 1 | Fret 2 | Fret 3 |
|--------|--------|--------|--------|--------|
| 6 (E) | E2 (40) | F2 (41) | F#2 (42) | G2 (43) |
| 5 (A) | A2 (45) | A#2 (46) | B2 (47) | C3 (48) |
| 4 (D) | D3 (50) | D#3 (51) | E3 (52) | F3 (53) |
| 3 (G) | G3 (55) | G#3 (56) | A3 (57) | A#3 (58) |
| 2 (B) | B3 (59) | C4 (60) | C#4 (61) | D4 (62) |
| 1 (E) | E4 (64) | F4 (65) | F#4 (66) | G4 (67) |

---

## WebSocket Support (Future)
Future versions may include WebSocket support for real-time transcription:
- Streaming audio input
- Real-time note detection
- Live chord progression updates

---

## SDK and Libraries

### Official Libraries
- **Python**: `guitar-transcription-client` (coming soon)
- **JavaScript**: `guitar-transcription-js` (coming soon)

### Community Libraries
- **Node.js**: Various community wrappers available
- **PHP**: Laravel package available
- **Ruby**: Ruby gem available

---

## Support and Contact
For API support, feature requests, or bug reports:
- Check the troubleshooting section in README.md
- Review this documentation
- Open an issue on GitHub

---

## Version History
- **v1.0.0**: Initial release with basic transcription features
- **v1.1.0**: Added chord detection and improved accuracy
- **v1.2.0**: Added tempo estimation and confidence scores

---

## Changelog
### v1.2.0 (Current)
- Added tempo estimation with confidence scoring
- Improved chord detection accuracy
- Added support for M4A format
- Enhanced error handling and validation

### v1.1.0
- Added chord progression detection
- Improved note timing accuracy
- Added confidence thresholds

### v1.0.0
- Initial release with melody transcription
- Basic file upload and processing
- RESTful API endpoints