# ✅ Use Python 3.10 for best prebuilt wheel coverage (numpy/scipy/madmom)
FROM python:3.10-slim

# --- System dependencies (runtime + light build toolchain) ---
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsndfile1 \
    build-essential \
    git \
 && rm -rf /var/lib/apt/lists/*

# --- App base ---
WORKDIR /app

# --- Install Python deps from requirements.txt first (layer caching) ---
# Copy requirements early for caching, but delay the full install until after madmom
COPY requirements.txt /app/requirements.txt

# Preinstall madmom and its strict build-time prerequisites in a controlled order
# to avoid pip build-isolation metadata-generation failures and dependency conflicts
RUN pip install --no-cache-dir --upgrade pip wheel setuptools \
 && pip install --no-cache-dir \
     cython==0.29.37 \
     numpy==1.23.5 \
     scipy==1.10.1 \
     mido==1.3.2 \
 && pip install --no-cache-dir madmom==0.16.1 \
 && pip install --no-cache-dir -r /app/requirements.txt

# --- Copy application source ---
COPY . /app

# --- Runtime env ---
ENV PORT=8080 \
    PYTHONUNBUFFERED=1

# (Optional for local runs; Railway doesn't require EXPOSE)
EXPOSE 8080

# --- Start FastAPI with Gunicorn + Uvicorn worker ---
# Shell form so $PORT expands (Railway provides it)
CMD ["sh", "-c", "gunicorn -k uvicorn.workers.UvicornWorker -w ${WORKERS:-1} -b 0.0.0.0:${PORT:-8080} main:app"]
