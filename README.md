# Guitar Music Helper - Audio Transcription API

A FastAPI-based web service that provides audio transcription capabilities for guitar music. This service analyzes audio files to extract musical information including melody notes, chord progressions, and tempo.

## 🎯 Overview

This application serves as a backend API for guitar music transcription, converting audio recordings into structured musical data. It uses machine learning models to identify notes, chords, and rhythm patterns from audio files.

## 🚀 Features

- **Audio File Upload**: Support for MP3, WAV, M4A, FLAC, and OGG formats
- **Large File Support**: Upload files up to 50MB for comprehensive audio analysis
- **Note Detection**: Identifies individual notes with timing, pitch, and confidence
- **Chord Recognition**: Extracts chord progressions from the audio
- **Tempo Estimation**: Calculates BPM (Beats Per Minute) with confidence
- **Real-time Processing**: Fast processing with progress tracking and memory optimization
- **RESTful API**: Clean, documented API endpoints with comprehensive error handling
- **Production Ready**: Deployed on Railway with automatic scaling and health monitoring

## 📁 Project Structure

```
guitar-music-helper-backend/
├── main.py                     # FastAPI application entry point
├── transcription_utils.py      # Audio processing and transcription utilities
├── models.py                  # Pydantic models for data validation
├── requirements.txt           # Python dependencies
├── .env                      # Environment configuration (local)
├── .env.example              # Environment variables template
├── .env.production           # Production environment configuration
├── .nixpacks.toml            # Railway deployment configuration
├── Procfile                  # Railway process configuration
├── RAILWAY_DEPLOYMENT_GUIDE.md # Deployment guide for Railway
└── README.md                 # This documentation file
```

## 🔧 Installation

### Prerequisites
- Python 3.8+
- pip (Python package manager)

### Setup Steps

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd guitar-music-helper-backend
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**
   ```bash
   cp .env.example .env
   # Edit .env with your configuration
   ```

5. **Run the application**
   ```bash
   uvicorn main:app --reload --host 0.0.0.0 --port 8000
   ```

## 📖 API Documentation

### Base URL
```
http://localhost:8000
```

### Endpoints

#### 1. **POST /transcribe**
Transcribe an audio file to extract musical information.

**Request:**
- **Method**: POST
- **Content-Type**: multipart/form-data
- **Max File Size**: 50MB
- **Supported Formats**: MP3, WAV, M4A, FLAC, OGG
- **Body**: 
  - `file`: Audio file (required)

**Response:**
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
      }
    ],
    "chords": [
      {
        "time": 0.0,
        "duration": 2.0,
        "chord": "C major",
        "confidence": 0.88
      }
    ],
    "tempo": {
      "bpm": 120.5,
      "confidence": 0.92
    }
  },
  "processingTime": 5.2
}
```

#### 2. **GET /health**
Health check endpoint to verify the service is running and dependencies are loaded.

**Response:**
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

#### 3. **GET /supported-formats**
Get supported file formats and size limits.

**Response:**
```json
{
  "supportedFormats": [".wav", ".mp3", ".m4a", ".flac", ".ogg"],
  "maxFileSizeMb": 50
}
```

#### 4. **GET /docs**
Interactive API documentation (Swagger UI)

#### 5. **GET /debug** (Development only)
Detailed debug information about the server state.

## 🎵 Data Models

### TranscriptionNote
Represents a single detected note in the audio.

| Field | Type | Description |
|-------|------|-------------|
| time | float | Start time in seconds |
| duration | float | Duration in seconds |
| pitch | int | MIDI note number (0-127) |
| frequency | float | Frequency in Hz |
| confidence | float | Detection confidence (0.0-1.0) |

### TranscriptionChord
Represents a detected chord.

| Field | Type | Description |
|-------|------|-------------|
| time | float | Start time in seconds |
| duration | float | Duration in seconds |
| chord | string | Chord symbol (e.g., "Cmaj", "Am") |
| confidence | float | Detection confidence (0.0-1.0) |

### TranscriptionTempo
Represents the detected tempo.

| Field | Type | Description |
|-------|------|-------------|
| bpm | float | Beats per minute |
| confidence | float | Detection confidence (0.0-1.0) |

## 🔍 Core Functions

### main.py Functions

#### `validate_file(file: UploadFile) -> None`
Validates uploaded audio files for format, size, and content.

**Parameters:**
- `file`: FastAPI UploadFile object

**Raises:**
- `HTTPException`: For invalid file formats or oversized files

#### `process_audio_file(file_path: str, min_confidence: float) -> Dict[str, Any]`
Processes audio file through the transcription pipeline.

**Parameters:**
- `file_path`: Path to the audio file
- `min_confidence`: Minimum confidence threshold for note detection

**Returns:**
- Dictionary containing melody, chords, and tempo data

### transcription_utils.py Functions

#### `midi_to_frequency(midi_note: int) -> float`
Converts MIDI note number to frequency in Hz.

**Formula**: `440.0 * (2.0 ** ((midi_note - 69) / 12.0))`

#### `frequency_to_midi(frequency: float) -> int`
Converts frequency in Hz to MIDI note number.

**Formula**: `int(round(12 * log2(frequency / 440.0) + 69))`

#### `estimate_tempo(audio: np.ndarray, sr: int) -> Tuple[float, float]`
Estimates tempo using librosa's beat tracking.

**Parameters:**
- `audio`: Audio time series as numpy array
- `sr`: Sample rate

**Returns:**
- Tuple of (tempo_bpm, confidence)

#### `extract_chords_from_notes(notes: List[Dict], time_step: float) -> List[Dict]`
Extracts chord progressions from note data using time-window analysis.

**Algorithm:**
1. Groups notes into time windows
2. Identifies unique pitches in each window
3. Generates basic chord symbols based on pitch combinations

#### `process_basic_pitch_output(model_output, midi_data, note_events, sample_rate, duration) -> Dict[str, Any]`
Converts basic-pitch model output into structured transcription format.

**Processing Steps:**
1. Extracts note events with timing and pitch
2. Converts to TranscriptionNote objects
3. Generates chord progressions
4. Returns structured data ready for API response

## ⚙️ Configuration

### Environment Variables
Create a `.env` file with:

```bash
# Server Configuration
ENVIRONMENT=development
LOG_LEVEL=INFO
PROCESSING_TIMEOUT=45

# File Upload Limits
MAX_FILE_SIZE_MB=50
ALLOWED_EXTENSIONS=.wav,.mp3,.m4a,.flac,.ogg

# Worker Configuration
MAX_WORKERS=1

# CORS Configuration (for frontend integration)
CORS_ORIGINS=["http://localhost:3000","http://localhost:5173","http://localhost:5174"]
BACKEND_URL=http://localhost:8000

# Debug Features
ENABLE_DEBUG_ENDPOINTS=true
```

### Production Configuration (Railway)
For production deployment on Railway, use:

```bash
# Production Environment
RAILWAY_ENVIRONMENT=production
ENVIRONMENT=production
LOG_LEVEL=WARNING
MAX_FILE_SIZE_MB=50
PROCESSING_TIMEOUT=60
ENABLE_DEBUG_ENDPOINTS=false
MAX_WORKERS=1
```

## 🧪 Testing

### Manual Testing
1. Start the server: `uvicorn main:app --reload`
2. Visit `http://localhost:8000/docs` for interactive API testing
3. Upload audio files and observe responses

### Automated Testing
```bash
pytest tests/ -v
```

## 🐛 Troubleshooting

### Common Issues

1. **"Module not found" errors**
   - Ensure virtual environment is activated
   - Run `pip install -r requirements.txt`

2. **"FFmpeg not found" errors**
   - Install FFmpeg system-wide
   - On Ubuntu: `sudo apt install ffmpeg`
   - On macOS: `brew install ffmpeg`
   - On Windows: Download from https://ffmpeg.org/

3. **Memory issues with large files**
   - Reduce `MAX_FILE_SIZE_MB` in configuration
   - Ensure sufficient RAM available

4. **Slow processing**
   - Check CPU usage
   - Reduce file size or quality
   - Adjust `MAX_WORKERS` for your system

## 📊 Performance Tips

1. **File Size**: Upload files up to 50MB for comprehensive analysis
2. **Audio Quality**: 22.05kHz sample rate is optimal for processing speed
3. **Format**: WAV and FLAC provide best accuracy, MP3 processes faster
4. **Length**: Files under 3 minutes process most efficiently
5. **Memory**: Large files (>20MB) may require additional processing time

## 🚀 Deployment

### Railway Deployment
This application is configured for deployment on Railway:

1. **Connect Repository**: Link your GitHub repository to Railway
2. **Environment Variables**: Set `MAX_FILE_SIZE_MB=50` in Railway dashboard
3. **Auto Deploy**: Railway will automatically deploy on git push
4. **Monitoring**: Use `/health` endpoint for health checks

See `RAILWAY_DEPLOYMENT_GUIDE.md` for detailed deployment instructions.

### Local Development
```bash
# Start development server
uvicorn main:app --reload --host 0.0.0.0 --port 8000

# With auto-reload for development
python -m uvicorn main:app --reload
```

## 🔒 Security Considerations

- File type validation prevents malicious uploads
- File size limits prevent DoS attacks
- Input sanitization on all parameters
- No file persistence after processing

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Support

For issues and questions:
- Check the troubleshooting section above
- Review API documentation at `/docs`
- Open an issue on GitHub
