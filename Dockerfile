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
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir --upgrade pip wheel setuptools \
 && pip install --no-cache-dir -r /app/requirements.txt

# --- Install madmom with its strict prerequisites (pin order matters) ---
RUN pip install --no-cache-dir \
      numpy==1.26.4 \
      scipy==1.10.1 \
      cython==0.29.37 \
      mido==1.3.2 \
 && pip install --no-cache-dir madmom==0.16.1

# --- Copy application source ---
COPY . /app

# --- Runtime env ---
ENV PORT=8080 \
    PYTHONUNBUFFERED=1

# (Optional for local runs; Railway doesn't require EXPOSE)
EXPOSE 8080

# --- Start FastAPI with Gunicorn + Uvicorn worker ---
CMD ["gunicorn", "-k", "uvicorn.workers.UvicornWorker", "-w", "1", "-b", "0.0.0.0:${PORT}", "main:app"]
