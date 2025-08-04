# Audio Transcription API Documentation

## API Overview

The Guitar Music Helper Audio Transcription API is a production-ready REST API that transcribes guitar audio files into structured musical data (chords, melody, tempo) using machine learning models.

**Base URL**: `https://your-api.railway.app` (or `http://localhost:8000` for development)

## Authentication

Currently, no authentication is required. The API uses IP-based rate limiting for abuse prevention.

## Rate Limits

| Endpoint | Limit |
|----------|--------|
| `/transcribe` | 5 requests/minute |
| `/transcribe-status` | 10 requests/minute |
| `/test-minimal-processing` | 3 requests/minute |
| `/test-dependencies` | 5 requests/minute |
| `/test-upload` | 5 requests/minute |
| Other endpoints | No specific limit |

Rate limit information is returned in response headers:
- `X-RateLimit-Remaining`: Requests remaining in current window
- `X-RateLimit-Reset`: Time when rate limit resets

## Request/Response Format

### Request Headers
```http
Content-Type: multipart/form-data
```

### Response Format
All responses follow this structure:
```json
{
  "success": true|false,
  "data": {...},           // Present on success
  "error": {...},          // Present on failure
  "processingTime": 1.23   // Processing time in seconds (some endpoints)
}
```

### Error Response Format
```json
{
  "success": false,
  "error": {
    "code": "ERROR_CODE",
    "message": "Human-readable message",
    "details": "Technical details",
    "request_id": "uuid-string"  // For debugging
  }
}
```

## File Requirements

### Supported Formats
- **Audio formats**: `.wav`, `.mp3`, `.m4a`, `.flac`, `.ogg`
- **Maximum file size**: 10MB (configurable)
- **Recommended**: Mono audio files for best results

### MIME Type Validation
The API validates both file extensions and MIME types:

| Extension | Allowed MIME Types |
|-----------|-------------------|
| `.wav` | `audio/wav`, `audio/x-wav` |
| `.mp3` | `audio/mpeg`, `audio/mp3`, `application/octet-stream` |
| `.m4a` | `audio/x-m4a`, `audio/m4a`, `audio/mp4`, `application/octet-stream` |
| `.flac` | `audio/flac`, `audio/x-flac`, `application/octet-stream` |
| `.ogg` | `audio/ogg`, `application/ogg`, `application/octet-stream` |

## Core Endpoints

### POST /transcribe

Transcribes an audio file into musical data (chords, melody, tempo).

**Parameters:**
- `file` (required): Audio file to transcribe

**Example Request:**
```bash
curl -X POST "https://your-api.railway.app/transcribe" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@guitar_sample.mp3"
```

**Success Response (200):**
```json
{
  "success": true,
  "data": {
    "metadata": {
      "filename": "guitar_sample.mp3",
      "duration": 30.5,
      "sampleRate": 22050,
      "processingTime": 12.34
    },
    "chords": [
      {
        "start": 0.0,
        "end": 2.5,
        "chord": "C",
        "confidence": 0.85
      },
      {
        "start": 2.5,
        "end": 5.0,
        "chord": "G",
        "confidence": 0.78
      }
    ],
    "melody": [
      {
        "start": 0.1,
        "end": 0.3,
        "pitch": 261.63,
        "note": "C4",
        "confidence": 0.92
      }
    ],
    "tempo": 120.0
  },
  "processingTime": 12.34
}
```

**Error Responses:**

**400 Bad Request** - Invalid file:
```json
{
  "success": false,
  "error": {
    "code": "REQUEST_ERROR",
    "message": "No file provided",
    "request_id": "abc123-def456"
  }
}
```

**413 Payload Too Large** - File too large:
```json
{
  "success": false,
  "error": {
    "code": "REQUEST_ERROR",
    "message": "File is too large (15.23MB). Maximum size is 10MB.",
    "request_id": "abc123-def456"
  }
}
```

**415 Unsupported Media Type** - Invalid format:
```json
{
  "success": false,
  "error": {
    "code": "REQUEST_ERROR",
    "message": "Unsupported file type '.txt'. Allowed types are: .wav, .mp3, .m4a, .flac, .ogg",
    "request_id": "abc123-def456"
  }
}
```

**422 Unprocessable Entity** - Processing failed:
```json
{
  "success": false,
  "error": {
    "code": "AUDIO_PROCESSING_ERROR",
    "message": "Failed to process the provided audio file.",
    "details": "Processing failed: Unable to decode audio",
    "request_id": "abc123-def456"
  }
}
```

**503 Service Unavailable** - Dependencies not loaded:
```json
{
  "success": false,
  "error": {
    "code": "SERVICE_UNAVAILABLE",
    "message": "The transcription service is temporarily unavailable.",
    "details": "Required ML dependencies or local models are not loaded.",
    "request_id": "abc123-def456"
  }
}
```

**507 Insufficient Storage** - Not enough memory:
```json
{
  "success": false,
  "error": {
    "code": "REQUEST_ERROR",
    "message": "Insufficient memory to process this file (8.5MB). Try a smaller file or try again later.",
    "request_id": "abc123-def456"
  }
}
```

### POST /transcribe-status

Pre-flight check to determine if a file can be transcribed without actually processing it.

**Parameters:**
- `file` (required): Audio file to analyze

**Example Request:**
```bash
curl -X POST "https://your-api.railway.app/transcribe-status" \
  -F "file=@large_audio.mp3"
```

**Success Response (200):**
```json
{
  "success": true,
  "file_info": {
    "filename": "large_audio.mp3",
    "size_mb": 8.5,
    "format": ".mp3"
  },
  "memory_analysis": {
    "estimated_memory_needed_mb": 34.0,
    "railway_memory_limit_mb": 500,
    "feasible": true,
    "recommendation": "Processing should be feasible"
  },
  "dependencies": {
    "dependencies_loaded": true,
    "models_loaded": true
  }
}
```

## Status and Information Endpoints

### GET /

Basic API information.

**Example Response (200):**
```json
{
  "service": "Guitar Music Helper Audio Transcription API",
  "version": "1.0.0",
  "status": "healthy"
}
```

### GET /health

Basic health check.

**Example Response (200):**
```json
{
  "status": "healthy",
  "dependencies_loaded": true,
  "models_loaded": true,
  "supported_formats": [".wav", ".mp3", ".m4a", ".flac", ".ogg"],
  "max_file_size_mb": 10,
  "timestamp": 1691234567.89
}
```

### GET /health/detailed

Comprehensive health check including system resources.

**Example Response (200):**
```json
{
  "status": "healthy",
  "checks": {
    "api": "healthy",
    "dependencies": true,
    "models": true,
    "memory": {
      "used_mb": 145.2,
      "psutil_available": true,
      "available_mb": 2048.5,
      "percent_used": 12.3
    },
    "executor": {
      "max_workers": 1
    },
    "disk": {
      "temp_dir_free_gb": 15.7
    }
  }
}
```

### GET /metrics

API usage statistics and performance metrics.

**Example Response (200):**
```json
{
  "requests": {
    "transcribe": 45,
    "transcribe-status": 12,
    "health": 8
  },
  "errors": {
    "transcribe:processing_error": 2,
    "transcribe:timeout": 1
  },
  "avg_processing_time_seconds": 8.45,
  "total_requests": 65,
  "total_errors": 3,
  "current_memory_usage_mb": 145.2
}
```

### GET /rate-limits

Information about API rate limits.

**Example Response (200):**
```json
{
  "endpoints": {
    "/transcribe": "5 requests per minute",
    "/test-minimal-processing": "3 requests per minute",
    "/test-dependencies": "5 requests per minute",
    "/transcribe-status": "10 requests per minute",
    "/test-upload": "5 requests per minute"
  },
  "rate_limit_headers": {
    "remaining": "X-RateLimit-Remaining",
    "reset": "X-RateLimit-Reset"
  },
  "note": "Rate limits are per IP address"
}
```

### GET /supported-formats

List of supported audio formats and file size limits.

**Example Response (200):**
```json
{
  "supportedFormats": [".wav", ".mp3", ".m4a", ".flac", ".ogg"],
  "maxFileSizeMb": 10
}
```

### GET /debug

Detailed system and dependency information (development only).

**Example Response (200):**
```json
{
  "python_version": "3.9.7 (default, Sep 16 2021, 16:59:28) ...",
  "platform": "Windows-10-10.0.19041-SP0",
  "dependencies_loaded": true,
  "models_loaded": true,
  "config": {
    "max_file_size_mb": 10,
    "max_workers": 1,
    "processing_timeout": 45,
    "environment": "development"
  },
  "supported_formats": [".wav", ".mp3", ".m4a", ".flac", ".ogg"],
  "ml_versions": {
    "librosa": "0.8.1",
    "numpy": "1.21.0"
  }
}
```

## Development and Testing Endpoints

These endpoints are only available in development mode:

### POST /test-upload

Tests file upload without processing.

**Example Response (200):**
```json
{
  "success": true,
  "message": "File uploaded successfully (no processing)",
  "filename": "test.mp3",
  "size_mb": 2.1,
  "upload_time": 0.15
}
```

### POST /test-dependencies

Tests ML dependency availability.

**Example Response (200):**
```json
{
  "success": true,
  "test_results": {
    "librosa": "OK - version 0.8.1",
    "numpy": "OK - version 1.21.0",
    "basic_pitch": "OK - predict function imported"
  },
  "dependencies_loaded": true,
  "models_loaded": true
}
```

### POST /test-minimal-processing

Step-by-step processing diagnostics.

**Example Response (200):**
```json
{
  "success": true,
  "step_results": {
    "file_upload": "OK - 2.1 MB",
    "librosa_load": "OK - 220500 samples at 22050Hz",
    "basic_pitch_predict": "OK - 45 note events"
  },
  "processing_time": 8.23,
  "message": "All steps completed successfully"
}
```

## SDK Examples

### JavaScript/TypeScript

```typescript
class AudioTranscriptionAPI {
  private baseUrl: string;

  constructor(baseUrl: string) {
    this.baseUrl = baseUrl;
  }

  async transcribe(file: File): Promise<TranscriptionResult> {
    const formData = new FormData();
    formData.append('file', file);

    const response = await fetch(`${this.baseUrl}/transcribe`, {
      method: 'POST',
      body: formData,
    });

    if (!response.ok) {
      const error = await response.json();
      throw new Error(`Transcription failed: ${error.error.message}`);
    }

    return response.json();
  }

  async checkStatus(file: File): Promise<StatusResult> {
    const formData = new FormData();
    formData.append('file', file);

    const response = await fetch(`${this.baseUrl}/transcribe-status`, {
      method: 'POST',
      body: formData,
    });

    return response.json();
  }
}

// Usage
const api = new AudioTranscriptionAPI('https://your-api.railway.app');
const result = await api.transcribe(audioFile);
console.log('Chords:', result.data.chords);
```

### Python

```python
import requests

class AudioTranscriptionAPI:
    def __init__(self, base_url: str):
        self.base_url = base_url

    def transcribe(self, file_path: str) -> dict:
        with open(file_path, 'rb') as file:
            files = {'file': file}
            response = requests.post(f'{self.base_url}/transcribe', files=files)
            response.raise_for_status()
            return response.json()

    def check_status(self, file_path: str) -> dict:
        with open(file_path, 'rb') as file:
            files = {'file': file}
            response = requests.post(f'{self.base_url}/transcribe-status', files=files)
            response.raise_for_status()
            return response.json()

# Usage
api = AudioTranscriptionAPI('https://your-api.railway.app')
result = api.transcribe('guitar_sample.mp3')
print(f"Found {len(result['data']['chords'])} chords")
```

## Best Practices

### 1. Error Handling

Always check the `success` field and handle errors appropriately:

```javascript
const response = await fetch('/transcribe', {
  method: 'POST',
  body: formData
});

const result = await response.json();

if (!result.success) {
  console.error('Transcription failed:', result.error.message);
  if (result.error.request_id) {
    console.log('Request ID for support:', result.error.request_id);
  }
  return;
}

// Process successful result
console.log('Transcription completed:', result.data);
```

### 2. Rate Limit Handling

Monitor rate limit headers and implement backoff:

```javascript
const response = await fetch('/transcribe', options);

const remaining = response.headers.get('X-RateLimit-Remaining');
const reset = response.headers.get('X-RateLimit-Reset');

if (remaining && parseInt(remaining) < 2) {
  console.warn('Approaching rate limit, consider waiting');
}

if (response.status === 429) {
  console.log('Rate limited, retry after:', reset);
  // Implement exponential backoff
}
```

This comprehensive documentation provides everything needed to understand and integrate with the production-ready Audio Transcription API.
