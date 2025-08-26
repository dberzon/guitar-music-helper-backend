# Railway Deployment Guide - 50MB File Upload Fix

## Problem
The backend was limiting file uploads to 10MB instead of the intended 50MB due to configuration issues.

## Changes Made

### 1. Updated Default Configuration
- Changed `main.py` line 149 from `Field(10, ...)` to `Field(50, ...)`
- This ensures the default value is 50MB if environment variables aren't loaded

### 2. Fixed nixpacks.toml
- Removed incorrect `server/` path references
- Now correctly references `requirements.txt` and `main:app`

### 3. Updated Environment Configuration
- Added explicit `ENVIRONMENT=production` to `.env.production`
- Ensured `MAX_FILE_SIZE_MB=50` is set in production

## Railway Deployment Steps

### Option 1: Environment Variables (Recommended)
Set these environment variables directly in Railway dashboard:
```
MAX_FILE_SIZE_MB=50
ENVIRONMENT=production
LOG_LEVEL=WARNING
PROCESSING_TIMEOUT=60
ENABLE_DEBUG_ENDPOINTS=false
```

### Option 2: Deploy with Updated Code
1. Commit the changes to git:
   ```bash
   git add .
   git commit -m "Fix: Increase file upload limit to 50MB"
   git push
   ```

2. Redeploy on Railway (it should auto-deploy if connected to git)

## Verification

After deployment, test the configuration:

```bash
# Check supported formats endpoint
curl https://web-production-84b20.up.railway.app/supported-formats

# Should return:
# {"supportedFormats": [".wav", ".mp3", ".m4a", ".flac", ".ogg"], "maxFileSizeMb": 50}
```

## Testing File Upload

Try uploading a file between 10-50MB to verify the fix:
- Files under 50MB should now work
- Files over 50MB should still be rejected with appropriate error

## Troubleshooting

If still getting 10MB limit:
1. Check Railway environment variables in dashboard
2. Check Railway logs for configuration loading messages
3. Use the `/debug` endpoint to see current config:
   ```bash
   curl https://web-production-84b20.up.railway.app/debug
   ```

## Memory Considerations

Note: 50MB files will require significant memory for processing:
- Estimated memory usage: ~200MB per file
- Railway Hobby plan has ~512MB total memory
- Consider upgrading Railway plan for consistent large file processing

---

## Madmom build quirks and Dockerfile recommendation

Madmom (used for chord recognition) runs build-time steps that require Cython
and a compatible NumPy to be present before madmom is built. On Railway the
default `pip install -r requirements.txt` can fail with:

```
ModuleNotFoundError: No module named 'Cython'
metadata-generation-failed
```

To avoid this, build madmom in a controlled order inside a Dockerfile where
you first install OS build tools and ffmpeg/libsndfile, then preinstall
Cython/NumPy/Scipy/mido, install madmom (we recommend installing from Git),
and finally install the rest of the requirements.

Add the following Dockerfile (or update your existing one) in
`guitar-music-helper-backend/Dockerfile`:

```dockerfile
FROM python:3.11-slim-bullseye

RUN apt-get update \
      && apt-get install -y --no-install-recommends \
          build-essential ffmpeg libsndfile1 git \
      && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt ./requirements.txt

RUN python -m venv --copies /opt/venv \
 && . /opt/venv/bin/activate \
 && pip install --upgrade pip setuptools wheel \
 && pip install "Cython<3" "numpy==1.26.4" "scipy==1.10.1" mido \
 && pip install --no-build-isolation "madmom @ git+https://github.com/CPJKU/madmom.git" \
 && pip install -r requirements.txt

ENV VIRTUAL_ENV=/opt/venv
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

COPY . /app
EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

Notes:
- We use `--no-build-isolation` so the preinstalled Cython/NumPy in the venv
   are visible to madmom's build step.
- Installing madmom from the Git repository is recommended because the PyPI
   release (0.16.1) is older and sometimes incompatible with newer Python
   or NumPy releases.

### Validation (after deploy)
After Railway builds the Docker image, open a shell in the container and run:

```bash
. /opt/venv/bin/activate
python -c "import madmom, numpy as np; print('madmom OK', getattr(madmom,'__version__','(no __version__)'), np.__version__)"
ffmpeg -version | head -n1
```

You should see `madmom OK` and an ffmpeg version line if the build succeeded.

