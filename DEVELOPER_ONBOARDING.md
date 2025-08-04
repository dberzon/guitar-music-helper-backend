# Developer Onboarding Guide

## Quick Start for New Developers

Welcome to the Guitar Music Helper Audio Transcription API! This guide will help you get up and running quickly.

## Prerequisites

### Required Knowledge
- Python 3.8+ 
- FastAPI framework basics
- RESTful API concepts
- Basic understanding of audio processing concepts

### System Requirements
- Python 3.8 or higher
- At least 2GB RAM for development
- 500MB free disk space

## Local Development Setup

### 1. Clone and Setup Environment

```bash
# Clone the repository
git clone https://github.com/dberzon/guitar-music-helper.git
cd guitar-music-helper/guitar-music-helper-backend

# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Environment Configuration

Create a `.env` file:
```env
# Development settings
RAILWAY_ENVIRONMENT=development
LOG_LEVEL=DEBUG
MAX_FILE_SIZE_MB=10
PROCESSING_TIMEOUT=45

# CORS settings for local development
CORS_ORIGINS=["http://localhost:5173", "http://127.0.0.1:5173"]
```

### 3. Start the Development Server

```bash
# Start with auto-reload
python main.py
# OR
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

### 4. Verify Installation

```bash
# Check API health
curl http://localhost:8000/health

# Check dependencies
curl http://localhost:8000/debug

# Access interactive docs
# Open browser: http://localhost:8000/docs
```

## Understanding the Codebase

### File Structure Overview

```
guitar-music-helper-backend/
├── main.py                 # Main FastAPI application
├── models.py              # Pydantic response models
├── transcription_utils.py # Audio processing utilities
├── requirements.txt       # Python dependencies
├── Procfile              # Railway deployment config
├── .env                  # Environment variables (local)
└── MAIN_PY_DOCUMENTATION.md # Detailed technical docs
```

### Key Architecture Patterns

#### 1. Dependency Injection Pattern
```python
# Instead of accessing global state
app.state.metrics.record_request()

# Use dependency injection
def get_metrics_collector(request: Request) -> SimpleMetrics:
    return request.app.state.metrics

async def endpoint(metrics: SimpleMetrics = Depends(get_metrics_collector)):
    metrics.record_request("endpoint_name")
```

#### 2. Context Manager Pattern
```python
# Automatic resource cleanup
async with temporary_audio_file(file) as tmp_path:
    # File is automatically cleaned up
    result = process_audio_file_sync(tmp_path)
```

#### 3. Error Handling Pattern
```python
try:
    # Processing logic
    pass
except SpecificError as e:
    metrics.record_error(endpoint_name, "specific_error")
    raise CustomError(f"Friendly message: {e}") from e
```

## Development Workflow

### 1. Making Changes

#### Adding a New Endpoint
```python
@app.post("/new-endpoint")
@limiter.limit("10/minute")  # Add rate limiting
async def new_endpoint(
    request: Request,
    file: UploadFile = Depends(validate_file),  # File validation
    metrics: SimpleMetrics = Depends(get_metrics_collector)  # Metrics
):
    """Endpoint description for OpenAPI docs"""
    metrics.record_request("new-endpoint")
    
    try:
        # Your logic here
        return {"success": True, "data": result}
    except Exception as e:
        metrics.record_error("new-endpoint", "processing_error")
        raise AudioProcessingError(f"Processing failed: {e}") from e
```

#### Update Rate Limits Documentation
```python
# In get_rate_limits() function
return {
    "endpoints": {
        "/transcribe": "5 requests per minute",
        "/new-endpoint": "10 requests per minute",  # Add your endpoint
        # ... other endpoints
    }
}
```

### 2. Testing Your Changes

#### Manual Testing
```bash
# Test health endpoints
curl http://localhost:8000/health
curl http://localhost:8000/health/detailed

# Test with a small audio file
curl -X POST "http://localhost:8000/transcribe" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@test_audio.mp3"

# Check metrics after testing
curl http://localhost:8000/metrics
```

#### Using the Interactive Docs
1. Open http://localhost:8000/docs
2. Try the endpoints with sample files
3. Check responses and error handling

### 3. Memory Testing

```bash
# Test with different file sizes
curl -X POST "http://localhost:8000/transcribe-status" \
     -F "file=@small_file.mp3"   # Should show feasible

curl -X POST "http://localhost:8000/transcribe-status" \
     -F "file=@large_file.mp3"   # May show not feasible
```

## Common Development Tasks

### 1. Adding New Configuration

```python
# In Config class
NEW_SETTING: int = Field(default=100, description="Description of setting")

@validator('NEW_SETTING')
def validate_new_setting(cls, v):
    if v < 1 or v > 1000:
        raise ValueError('NEW_SETTING must be between 1 and 1000')
    return v
```

### 2. Adding New Error Types

```python
# Define custom exception
class NewProcessingError(Exception):
    """Exception for new processing type."""
    pass

# Add exception handler
@app.exception_handler(NewProcessingError)
async def new_processing_error_handler(request: Request, exc: NewProcessingError):
    request_id = getattr(request.state, 'request_id', None)
    return JSONResponse(
        status_code=422,
        content={
            "success": False,
            "error": {
                "code": "NEW_PROCESSING_ERROR",
                "message": "Friendly error message",
                "details": str(exc),
                "request_id": request_id
            },
        },
    )
```

### 3. Adding New Metrics

```python
# In SimpleMetrics class
def record_custom_metric(self, metric_name: str, value: float):
    """Record a custom metric"""
    self.custom_metrics[metric_name] = value

# In get_stats method
return {
    # ... existing stats
    "custom_metrics": dict(self.custom_metrics)
}
```

## Debugging Guide

### 1. Common Issues and Solutions

#### "Dependencies not loaded"
```bash
# Check what's missing
curl http://localhost:8000/test-dependencies

# Install missing packages
pip install librosa basic-pitch numpy
```

#### "Memory errors during processing"
```bash
# Check current memory usage
curl http://localhost:8000/health/detailed

# Test with smaller files first
curl -X POST "http://localhost:8000/test-minimal-processing" \
     -F "file=@small_test.mp3"
```

#### "CORS errors from frontend"
```python
# Check CORS configuration in Config class
CORS_ORIGINS: list[str] = [
    "http://localhost:3000",  # Add your frontend URL
    "http://localhost:5173",
]
```

### 2. Logging and Monitoring

#### Enable Debug Logging
```env
LOG_LEVEL=DEBUG
```

#### Monitor Requests
```bash
# Watch logs while testing
tail -f app.log  # If logging to file

# Check metrics periodically
watch -n 5 'curl -s http://localhost:8000/metrics | jq'
```

#### Request Tracing
Every request gets a unique ID. Look for it in logs:
```
Audio processing error for request http://localhost:8000/transcribe (ID: abc123-def456): Processing failed
```

## Code Style and Standards

### 1. Coding Standards

#### Function Documentation
```python
def process_audio_file(file_path: str, max_duration: float = None) -> dict:
    """
    Process an audio file and extract musical information.
    
    Args:
        file_path: Path to the audio file
        max_duration: Maximum duration to process (seconds)
        
    Returns:
        Dictionary containing chords, melody, and metadata
        
    Raises:
        AudioProcessingError: If processing fails
    """
```

#### Error Handling
```python
# Always provide context in error messages
try:
    result = process_file(path)
except Exception as e:
    logger.error(f"Failed to process {path}: {e}", exc_info=True)
    raise AudioProcessingError(f"Processing failed for {path}: {e}") from e
```

#### Memory Management
```python
import gc

def memory_intensive_function():
    try:
        # Do work
        large_data = load_large_dataset()
        result = process(large_data)
        
        # Clean up explicitly
        del large_data
        gc.collect()
        
        return result
    except Exception as e:
        # Clean up on error too
        gc.collect()
        raise
```

### 2. Testing Guidelines

#### Unit Test Example
```python
import pytest
from unittest.mock import Mock, patch

def test_memory_check():
    """Test memory availability checking"""
    # Mock psutil unavailable
    with patch('main.PSUTIL_AVAILABLE', False):
        assert check_memory_availability(100) == True  # Should assume OK
    
    # Mock sufficient memory
    with patch('main.psutil') as mock_psutil:
        mock_psutil.virtual_memory.return_value.available = 1024 * 1024 * 1024  # 1GB
        assert check_memory_availability(100) == True  # 100MB request should be OK
```

## Deployment

### 1. Railway Deployment

The app is configured for Railway with:
- Single worker Gunicorn configuration
- Environment-based settings
- Memory-optimized processing

```bash
# Deploy to Railway
railway up

# Check deployment health
curl https://your-app.railway.app/health
```

### 2. Environment Variables for Production

```env
RAILWAY_ENVIRONMENT=production
LOG_LEVEL=WARNING
MAX_FILE_SIZE_MB=10
PROCESSING_TIMEOUT=60
```

## Getting Help

### 1. Documentation Resources
- `MAIN_PY_DOCUMENTATION.md` - Detailed technical documentation
- FastAPI docs: https://fastapi.tiangolo.com/
- Basic-pitch docs: https://github.com/spotify/basic-pitch

### 2. Debugging Tools
- `/debug` endpoint - System information
- `/health/detailed` - Comprehensive health check
- `/metrics` - API usage statistics
- `/test-*` endpoints - Diagnostic tools

### 3. Common Commands Reference

```bash
# Development
python main.py                    # Start development server
pip freeze > requirements.txt     # Update dependencies

# Testing
curl http://localhost:8000/health # Basic health check
curl http://localhost:8000/docs   # API documentation

# Deployment
railway up                        # Deploy to Railway
railway logs                      # View deployment logs
```

Welcome to the team! The codebase is designed to be maintainable and well-documented. Don't hesitate to explore the diagnostic endpoints and use the comprehensive error handling to understand how everything works.
