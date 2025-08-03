# Frontend Integration Testing Guide

## Overview
This directory contains tools to test the API endpoints and ensure frontend compatibility with the Guitar Music Helper backend.

## Files
- `test_frontend_integration.html` - Browser-based testing interface
- `test_api_integration.py` - Python script for automated API testing
- `TESTING_README.md` - This documentation

## Quick Start

### 1. Start the Backend Server
```bash
# From the project root directory
python main.py
```

### 2. Test API Endpoints

#### Option A: Browser Testing
1. Open `test_frontend_integration.html` in your browser
2. Click the test buttons to verify each endpoint
3. Check the response sections for detailed results

#### Option B: Python Testing
```bash
# Run the automated test suite
python test_api_integration.py

# Test against a different server
python test_api_integration.py http://your-server:8000
```

### 3. Verify Frontend Integration

The frontend JavaScript error mentioned in the feedback was:
```
TypeError: Cannot read properties of undefined (reading 'map')
```

This occurs when the `/supported-formats` endpoint doesn't return the expected JSON structure. The test tools will verify:

1. **Response Format**: Ensure the endpoint returns `{"supportedFormats": [...]}`
2. **Data Type**: Ensure `supportedFormats` is an array
3. **Content**: Ensure the array contains valid format strings

## Expected API Responses

### GET /
```json
{
  "message": "Guitar Music Helper Audio Transcription API",
  "version": "1.0.0",
  "status": "running"
}
```

### GET /health
```json
{
  "status": "healthy",
  "timestamp": "2025-08-03T11:31:55.160Z"
}
```

### GET /supported-formats
```json
{
  "supportedFormats": ["mp3", "wav", "flac", "m4a", "ogg"]
}
```

## Troubleshooting

### Common Issues

1. **CORS Errors**: Ensure the backend has CORS enabled for your frontend domain
2. **Network Issues**: Check if the backend is running on the expected port
3. **Response Format**: Verify the JSON structure matches frontend expectations

### Debug Steps

1. **Check Backend Logs**: Look for any error messages in the console
2. **Test with curl**:
   ```bash
   curl http://localhost:8000/supported-formats
   ```
3. **Use Browser DevTools**: Check the Network tab for request/response details

## Integration Checklist

- [ ] Backend server is running
- [ ] All API endpoints return 200 status
- [ ] `/supported-formats` returns valid JSON array
- [ ] Frontend can access endpoints without CORS issues
- [ ] Response times are acceptable (< 1s for simple endpoints)

## Next Steps

After successful testing:
1. Update frontend code to handle the response format
2. Add proper error handling for network failures
3. Implement loading states for API calls
4. Add retry logic for failed requests