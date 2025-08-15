# Developer Guide - Guitar Music Helper

## Overview
This guide is designed for developers who want to understand, contribute to, or extend the Guitar Music Helper Audio Transcription API. It covers the codebase architecture, development setup, testing strategies, and contribution guidelines.

---

## Architecture Overview

### System Architecture
```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│   Client App    │────▶│   FastAPI        │────▶│   Audio         │
│   (Web/Mobile)  │     │   Server         │     │   Processing    │
└─────────────────┘     └──────────────────┘     └─────────────────┘
                                │                          │
                                ▼                          ▼
                        ┌──────────────────┐     ┌─────────────────┐
                        │   File Storage   │     │   ML Models     │
                        │   (Temp Files)   │     │   (Basic Pitch) │
                        └──────────────────┘     └─────────────────┘
```

### Component Breakdown
- **FastAPI Server** ([`main.py`](main.py)): HTTP request handling, validation, and response formatting
- **Audio Processing** ([`transcription_utils.py`](transcription_utils.py)): Core audio analysis and transcription logic
- **ML Integration**: Basic Pitch model for note detection
- **File Management**: Temporary file handling and cleanup

---

## Development Setup

### Prerequisites
```bash
# System requirements
Python 3.8+
FFmpeg (system-wide installation)
8GB+ RAM recommended for large files

# Development tools
pip install -r requirements.txt
pip install pytest pytest-asyncio black flake8 mypy
```

### Local Development
```bash
# 1. Clone and setup
git clone <repository-url>
cd guitar-music-helper-backend
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Install development dependencies
pip install -r requirements-dev.txt

# 4. Run tests
pytest tests/ -v

# 5. Start development server
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### Environment Setup
Create `.env.development`:
```bash
# Development settings
DEBUG=true
LOG_LEVEL=DEBUG
MAX_FILE_SIZE_MB=50  # Smaller for development
ALLOWED_EXTENSIONS=mp3,wav
```

---

## Code Structure Deep Dive

### File Organization
```
guitar-music-helper-backend/
├── main.py                 # FastAPI application
├── transcription_utils.py  # Audio processing utilities
├── models.py              # Pydantic models
├── config.py              # Configuration management
├── tests/                 # Test suite
│   ├── test_main.py
│   ├── test_transcription.py
│   └── fixtures/
├── docs/                  # Documentation
│   ├── API_DOCUMENTATION.md
│   └── DEVELOPER_GUIDE.md
└── scripts/               # Utility scripts
    ├── setup_dev.sh
    └── run_tests.sh
```

### Core Modules

#### 1. main.py - FastAPI Application
**Responsibilities:**
- HTTP endpoint definitions
- Request validation
- File upload handling
- Error handling and responses
- API documentation generation

**Key Functions:**
- `validate_file()`: File format and size validation
- `process_audio_file()`: Main processing pipeline
- Exception handlers for various error types

#### 2. transcription_utils.py - Audio Processing
**Responsibilities:**
- Audio file loading and preprocessing
- Note detection using Basic Pitch
- Chord extraction from note data
- Tempo estimation
- Data formatting and cleanup

**Key Functions:**
- `estimate_tempo()`: BPM detection using librosa
- `extract_chords_from_notes()`: Chord progression extraction
- `process_basic_pitch_output()`: ML output processing

---

## Testing Strategy

### Test Categories

#### 1. Unit Tests
```python
# test_transcription_utils.py
def test_midi_to_frequency():
    """Test MIDI to frequency conversion accuracy"""
    assert midi_to_frequency(69) == 440.0
    assert abs(midi_to_frequency(60) - 261.63) < 0.01

def test_chord_extraction():
    """Test chord detection from note sequences"""
    notes = [
        {"time": 0.0, "pitch": 60, "duration": 1.0},
        {"time": 0.0, "pitch": 64, "duration": 1.0},
        {"time": 0.0, "pitch": 67, "duration": 1.0}
    ]
    chords = extract_chords_from_notes(notes, 0.5)
    assert chords[0]["chord"] == "Cmaj"
```

#### 2. Integration Tests
```python
# test_main.py
def test_audio_transcription_endpoint():
    """Test complete transcription workflow"""
    with open("test_guitar.mp3", "rb") as f:
        response = client.post("/transcribe", files={"file": f})
    assert response.status_code == 200
    data = response.json()
    assert "melody" in data
    assert "chords" in data
    assert "tempo" in data
```

#### 3. Performance Tests
```python
# test_performance.py
def test_large_file_processing():
    """Test processing of large audio files"""
    large_file = generate_large_test_file(50 * 1024 * 1024)  # 50MB
    start_time = time.time()
    response = client.post("/transcribe", files={"file": large_file})
    processing_time = time.time() - start_time
    assert processing_time < 60  # Should complete within 60 seconds
```

### Test Data
Create test fixtures in `tests/fixtures/`:
- `guitar_c_major.mp3`: Simple C major scale
- `guitar_chords.mp3`: Common chord progression
- `guitar_empty.mp3`: Silent audio for edge cases
- `guitar_noise.mp3`: Background noise only

### Running Tests
```bash
# Run all tests
pytest tests/ -v

# Run specific test categories
pytest tests/test_main.py -v
pytest tests/test_transcription.py -v

# Run with coverage
pytest tests/ --cov=main --cov=transcription_utils --cov-report=html

# Run performance tests only
pytest tests/test_performance.py -v -m performance
```

---

## Development Workflow

### Code Style
```bash
# Format code
black main.py transcription_utils.py

# Lint code
flake8 main.py transcription_utils.py

# Type checking
mypy main.py transcription_utils.py
```

### Pre-commit Hooks
Create `.pre-commit-config.yaml`:
```yaml
repos:
  - repo: https://github.com/psf/black
    rev: 22.3.0
    hooks:
      - id: black
  - repo: https://github.com/pycqa/flake8
    rev: 4.0.1
    hooks:
      - id: flake8
```

### Git Workflow
```bash
# Feature branch workflow
git checkout -b feature/improve-chord-detection
# Make changes
git add .
git commit -m "feat: improve chord detection accuracy"
git push origin feature/improve-chord-detection
# Create pull request
```

---

## Extending the Application

### Adding New Features

#### 1. New Audio Format Support
```python
# In main.py
def validate_file(file: UploadFile) -> None:
    allowed_extensions = {"mp3", "wav", "m4a", "flac", "ogg"}  # Add ogg
    # ... rest of validation
```

#### 2. New Transcription Features
```python
# In transcription_utils.py
def detect_key_signature(audio: np.ndarray, sr: int) -> Dict[str, Any]:
    """Detect the key signature of the audio"""
    # Implementation here
    return {"key": "C major", "confidence": 0.85}

# In main.py
# Add to response
response["key_signature"] = detect_key_signature(audio, sr)
```

#### 3. New API Endpoints
```python
# In main.py
@app.post("/analyze/tuning")
async def analyze_tuning(file: UploadFile = File(...)):
    """Analyze if the guitar is in tune"""
    # Implementation
    return {"tuning": "standard", "deviation": 0.0}
```

### Plugin Architecture
Consider implementing a plugin system:
```python
# plugins/base.py
class TranscriptionPlugin:
    def process(self, audio: np.ndarray, sr: int) -> Dict[str, Any]:
        raise NotImplementedError

# plugins/chord_extensions.py
class ExtendedChordPlugin(TranscriptionPlugin):
    def process(self, audio: np.ndarray, sr: int) -> Dict[str, Any]:
        # Advanced chord detection
        return {"extended_chords": [...]}
```

---

## Performance Optimization

### Profiling
```bash
# Profile memory usage
python -m memory_profiler main.py

# Profile CPU usage
python -m cProfile -o profile.stats main.py
```

### Optimization Strategies

#### 1. Memory Optimization
```python
# Use generators for large files
def process_audio_streaming(file_path: str):
    """Process audio in chunks to reduce memory usage"""
    # Implementation
```

#### 2. Caching
```python
from functools import lru_cache

@lru_cache(maxsize=128)
def get_chord_pattern(notes: Tuple[int, ...]) -> str:
    """Cache chord pattern lookups"""
    # Implementation
```

#### 3. Parallel Processing
```python
from concurrent.futures import ProcessPoolExecutor

def process_multiple_files(file_paths: List[str]):
    """Process multiple files in parallel"""
    with ProcessPoolExecutor() as executor:
        results = executor.map(process_single_file, file_paths)
```

---

## Deployment

### Docker Deployment
```dockerfile
# Dockerfile
FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Production Considerations
- Use Gunicorn with Uvicorn workers
- Implement proper logging
- Set up monitoring and alerting
- Use environment variables for configuration
- Implement rate limiting
- Set up proper file cleanup

### Environment Variables
```bash
# Production settings
DEBUG=false
LOG_LEVEL=INFO
MAX_WORKERS=4
MAX_FILE_SIZE_MB=50
REDIS_URL=redis://localhost:6379
```

---

## Monitoring and Logging

### Structured Logging
```python
import logging
import json

logger = logging.getLogger(__name__)

def log_transcription_request(file_size: int, processing_time: float):
    logger.info(json.dumps({
        "event": "transcription_completed",
        "file_size_mb": file_size / 1024 / 1024,
        "processing_time": processing_time,
        "timestamp": datetime.utcnow().isoformat()
    }))
```

### Metrics Collection
```python
# Using prometheus_client
from prometheus_client import Counter, Histogram

transcription_requests = Counter('transcription_requests_total', 'Total transcription requests')
processing_duration = Histogram('transcription_processing_seconds', 'Processing time')
```

---

## Security Considerations

### File Upload Security
```python
def validate_file_security(file: UploadFile) -> bool:
    """Additional security checks for uploaded files"""
    # Check file signature
    # Scan for malware
    # Validate file structure
    return True
```

### Rate Limiting
```python
from slowapi import Limiter

limiter = Limiter(key_func=lambda: get_remote_address())

@app.post("/transcribe")
@limiter.limit("10/minute")
async def transcribe_audio(...):
    # Implementation
```

---

## Troubleshooting Guide

### Common Issues

#### 1. Memory Issues
**Symptoms**: Process killed, slow processing
**Solutions**:
- Reduce file size limits
- Implement streaming processing
- Add memory monitoring

#### 2. Audio Processing Failures
**Symptoms**: 422 errors, corrupted output
**Solutions**:
- Validate audio file integrity
- Check FFmpeg installation
- Test with known good files

#### 3. Model Loading Issues
**Symptoms**: Slow startup, model not found
**Solutions**:
- Pre-download models in Docker
- Implement model caching
- Add health checks

### Debug Mode
```bash
# Enable debug logging
export DEBUG=true
export LOG_LEVEL=DEBUG

# Run with debug server
uvicorn main:app --reload --log-level debug
```

---

## Contributing Guidelines

### Code Review Checklist
- [ ] Tests added for new functionality
- [ ] Documentation updated
- [ ] Code follows style guidelines
- [ ] Performance impact considered
- [ ] Security implications reviewed
- [ ] Backward compatibility maintained

### Pull Request Template
```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Documentation update

## Testing
- [ ] Unit tests pass
- [ ] Integration tests pass
- [ ] Manual testing completed

## Performance Impact
Describe any performance implications
```

---

## Resources and References

### Libraries Used
- **FastAPI**: Web framework
- **librosa**: Audio analysis
- **basic-pitch**: Note detection ML model
- **pydantic**: Data validation

### Learning Resources
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [librosa Documentation](https://librosa.org/doc/latest/index.html)
- [Audio Signal Processing](https://www.audiolabs-erlangen.de/resources/MIR/books)

### Community
- GitHub Issues: Bug reports and feature requests
- Discussions: Architecture decisions and questions
- Wiki: Additional documentation and examples