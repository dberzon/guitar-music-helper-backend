# Changelog

All notable changes to the Guitar Music Helper Audio Transcription API will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.3.0] - 2025-08-15

### Added
- **Increased file upload limit to 50MB** (previously 10MB)
- Added support for OGG audio format
- Added `/supported-formats` endpoint to query current limits and formats
- Added comprehensive health checks with dependency status
- Added Railway cloud deployment configuration
- Added memory monitoring and optimization
- Added detailed debug endpoints for development
- Added environment-based configuration management
- Added CORS configuration for frontend integration
- Added rate limiting on API endpoints
- Added comprehensive logging and telemetry

### Changed
- Updated default configuration to use 50MB file limit
- Improved error handling and validation
- Enhanced processing efficiency for large files
- Optimized memory usage during audio processing
- Updated API response format to include success status and metadata
- Changed chord format from abbreviated (e.g., "Cmaj") to full names (e.g., "C major")
- Improved processing timeout handling for large files

### Fixed
- Fixed Railway deployment configuration paths in `.nixpacks.toml`
- Fixed environment variable loading priority
- Fixed memory management issues with large audio files
- Fixed CORS configuration for production deployment

### Technical
- Migrated to Railway cloud platform for production hosting
- Added comprehensive environment variable configuration
- Implemented circuit breaker pattern for fault tolerance
- Added metrics collection and monitoring
- Enhanced documentation with deployment guides

## [1.2.0] - 2024-12-15

### Added
- Tempo estimation with confidence scoring
- Support for M4A audio format
- Enhanced error handling and validation
- Processing timeout configuration
- Confidence scoring for all detection types

### Changed
- Improved chord detection accuracy
- Enhanced note timing precision
- Better handling of audio file formats

### Fixed
- Memory leaks during processing
- Timing accuracy issues with certain file types

## [1.1.0] - 2024-11-20

### Added
- Chord progression detection
- Confidence thresholds for note filtering
- Support for FLAC audio format
- Basic tempo detection

### Changed
- Improved note timing accuracy
- Enhanced API response structure
- Better error messages

### Fixed
- File upload validation issues
- Processing errors with certain MP3 files

## [1.0.0] - 2024-10-30

### Added
- Initial release with melody transcription
- Basic file upload and processing
- RESTful API endpoints
- Support for MP3 and WAV formats
- Note detection with pitch and timing
- FastAPI-based web service
- Interactive API documentation (Swagger UI)

### Features
- Audio file upload validation
- Note-by-note transcription
- MIDI pitch number mapping
- Frequency analysis
- Basic error handling

---

## Migration Guide

### Upgrading to v1.3.0

#### API Response Format Changes
The API response format has changed to include a success wrapper:

**Before (v1.2.0):**
```json
{
  "melody": [...],
  "chords": [...],
  "tempo": {...}
}
```

**After (v1.3.0):**
```json
{
  "success": true,
  "data": {
    "metadata": {...},
    "melody": [...],
    "chords": [...],
    "tempo": {...}
  },
  "processingTime": 5.2
}
```

#### Chord Format Changes
Chord names now use full descriptive names:

**Before:** `"Cmaj"`, `"Am"`, `"G7"`
**After:** `"C major"`, `"A minor"`, `"G7"`

#### File Size Limit
The maximum file size has been increased from 10MB to 50MB. Update your client-side validation accordingly.

#### New Endpoints
- `/supported-formats` - Get current format and size limits
- `/health` - Enhanced health check with dependency status

### Upgrading to v1.2.0

#### New Response Fields
- All detections now include `confidence` scores
- Tempo estimation added to response
- Enhanced metadata in responses

### Upgrading to v1.1.0

#### New Features
- Chord detection results added to API response
- Confidence filtering options
- FLAC format support

---

## Deprecation Notices

### v1.3.0
- The `min_confidence` parameter has been removed from the `/transcribe` endpoint
- Old chord naming format (abbreviated) is deprecated in favor of full names

### v1.2.0
- Simple health check response format deprecated in favor of detailed status

---

## Breaking Changes

### v1.3.0
- API response format changed to include success wrapper and metadata
- Chord naming convention changed from abbreviated to full names
- Minimum supported file processing changed due to memory optimizations

### v1.1.0
- Response structure expanded to include chord data
- Some MIME type validations became stricter

---

## Security Updates

### v1.3.0
- Enhanced file validation and type checking
- Improved CORS configuration
- Rate limiting implementation
- Memory usage monitoring to prevent DoS

### v1.2.0
- Enhanced file upload validation
- Improved error message sanitization

---

## Performance Improvements

### v1.3.0
- Optimized memory usage for large file processing
- Improved processing efficiency by up to 40%
- Better garbage collection during audio processing
- Reduced memory footprint for Railway deployment

### v1.2.0
- Faster chord detection algorithms
- Reduced processing time for longer audio files

### v1.1.0
- Improved audio loading performance
- Better memory management

---

## Known Issues

### v1.3.0
- Very large files (40-50MB) may occasionally timeout on slower networks
- Some OGG files with unusual encoding may not be processed correctly
- Railway deployment may have cold start delays for the first request

### v1.2.0
- Tempo detection may be inaccurate for files with irregular timing
- M4A files with DRM protection are not supported

### v1.1.0
- Chord detection accuracy varies with audio quality
- FLAC files larger than 20MB may process slowly

---

## Acknowledgments

- Railway team for hosting platform
- basic-pitch library contributors
- librosa and audio processing community
- Beta testers and early adopters
