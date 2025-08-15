# API Documentation - Guitar Music Helper

## Overview
This document provides detailed API documentation for the Guitar Music Helper Audio Transcription API. It includes endpoint specifications, request/response formats, error handling, and usage examples.

---

## Base URL
```
Production: https://web-production-84b20.up.railway.app
Development: http://localhost:8000
```

## Authentication
This API does not require authentication for basic usage.

---

## Endpoints

### 1. Health Check
Check if the API service is running and dependencies are loaded.

**Endpoint:** `GET /health`

**Description:** Returns the current health status of the service and its dependencies.

**Request:**
```
GET /health
```

**Response (200 OK):**
```json
{
  "status": "healthy",
  "dependencies_loaded": true,
  "models_loaded": true,
  "supported_formats": [".wav", ".mp3", ".m4a", ".flac", ".ogg"],
  "max_file_size_mb": 50,
  "timestamp": 1692089400.123
}
```

**Response Fields:**
- `status` (string): "healthy", "unhealthy", or "degraded"
- `dependencies_loaded` (boolean): Whether ML dependencies are loaded
- `models_loaded` (boolean): Whether transcription models are available
- `supported_formats` (array): List of supported file extensions
- `max_file_size_mb` (number): Maximum file size in megabytes
- `timestamp` (number): Unix timestamp of the health check

---

### 2. Get Supported Formats
Get information about supported file formats and size limits.

**Endpoint:** `GET /supported-formats`

**Description:** Returns the list of supported audio formats and current file size limits.

**Request:**
```
GET /supported-formats
```

**Response (200 OK):**
```json
{
  "supportedFormats": [".wav", ".mp3", ".m4a", ".flac", ".ogg"],
  "maxFileSizeMb": 50
}
```

**Response Fields:**
- `supportedFormats` (array): List of supported file extensions
- `maxFileSizeMb` (number): Maximum file size in megabytes

---

### 3. Transcribe Audio
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
| file | file | Yes | Audio file (MP3, WAV, M4A, FLAC, OGG) |

**File Requirements:**
- **Supported Formats:** MP3, WAV, M4A, FLAC, OGG
- **Maximum Size:** 50MB (increased from previous 10MB limit)
- **Recommended Quality:** 22.05kHz sample rate for optimal processing
- **Maximum Duration:** Up to 5 minutes for reliable processing

**Response (200 OK):**
```json
{
  "success": true,
  "data": {
    "metadata": {
      "filename": "guitar-sample.mp3",
      "duration": 180.5,
      "sampleRate": 22050,
      "processingTime": 5.2
    },
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
        "chord": "C major",
        "confidence": 0.92
      },
      {
        "time": 2.0,
        "duration": 2.0,
        "chord": "G major",
        "confidence": 0.85
      }
    ],
    "tempo": {
      "bpm": 120.5,
      "confidence": 0.94
    }
  },
  "processingTime": 5.2
}
```

**Response Fields:**

#### Success Response Structure
- `success` (boolean): Always true for successful transcriptions
- `data` (object): Contains all transcription results
- `processingTime` (float): Total processing time in seconds

#### Metadata Object
- `filename` (string): Original filename of the uploaded audio
- `duration` (float): Duration of the audio in seconds
- `sampleRate` (number): Sample rate used for processing (typically 22050 Hz)
- `processingTime` (float): Time taken to process the audio in seconds

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
- `chord` (string): Chord name (e.g., "C major", "A minor", "G major")
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
curl -X POST "https://web-production-84b20.up.railway.app/transcribe" \
  -F "file=@guitar_sample.mp3"

# Local development
curl -X POST "http://localhost:8000/transcribe" \
  -F "file=@guitar_sample.mp3"

# Check supported formats
curl "https://web-production-84b20.up.railway.app/supported-formats"
```

### Python Example
```python
import requests

# Production URL
url = "https://web-production-84b20.up.railway.app/transcribe"

# Basic usage
files = {"file": open("guitar_sample.mp3", "rb")}
response = requests.post(url, files=files)
data = response.json()

# Check if successful
if data["success"]:
    transcription = data["data"]
    melody_notes = transcription["melody"]
    chords = transcription["chords"]
    tempo = transcription["tempo"]["bpm"]
    processing_time = data["processingTime"]
    
    print(f"Processed in {processing_time:.2f} seconds")
    print(f"Found {len(melody_notes)} notes and {len(chords)} chords")
    print(f"Tempo: {tempo:.1f} BPM")
```

### JavaScript Example
```javascript
// Using fetch API with production URL
const formData = new FormData();
formData.append('file', audioFile);

fetch('https://web-production-84b20.up.railway.app/transcribe', {
  method: 'POST',
  body: formData
})
.then(response => response.json())
.then(data => {
  if (data.success) {
    const transcription = data.data;
    console.log('Melody:', transcription.melody);
    console.log('Chords:', transcription.chords);
    console.log('Tempo:', transcription.tempo.bpm);
    console.log('Processing time:', data.processingTime);
  }
})
.catch(error => {
  console.error('Error:', error);
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
| File Size | Duration | Processing Time | Memory Usage |
|-----------|----------|-----------------|--------------|
| < 1MB | < 10s | 1-3 seconds | ~50MB |
| 1-5MB | 10-30s | 3-8 seconds | ~100MB |
| 5-20MB | 30s-2min | 8-20 seconds | ~200MB |
| 20-50MB | 2-5min | 20-60 seconds | ~400MB |

**Note:** Processing times may vary based on server load and audio complexity.

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
- **v1.3.0**: Increased file upload limit to 50MB, added OGG support, Railway deployment
- **v1.2.0**: Added tempo estimation and confidence scores
- **v1.1.0**: Added chord detection and improved accuracy
- **v1.0.0**: Initial release with basic transcription features

---

## Changelog
### v1.3.0 (Current)
- **Increased file upload limit to 50MB** (previously 10MB)
- Added support for OGG audio format
- Deployed to Railway cloud platform for production use
- Enhanced error handling and memory management
- Added `/supported-formats` endpoint
- Improved processing efficiency for large files
- Added comprehensive health checks and monitoring

### v1.2.0
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