# Guitar Music Helper — Backend API

This is the backend service for the **Guitar Music Helper** application.  
It provides audio-to-notation transcription for guitar music using [FastAPI](https://fastapi.tiangolo.com/) and [Spotify's Basic Pitch](https://github.com/spotify/basic-pitch).

### 🎯 Features

- 🎸 Upload `.wav`, `.mp3`, `.ogg`, `.flac`, and `.m4a` files
- 🧠 Transcribes audio to melody + chord data using `basic-pitch`
- 🧾 Returns time-aligned transcription with tempo + metadata
- ⚡ FastAPI + Uvicorn + CORS for modern frontend integration
- 📦 Deployable to Railway, Replit, Render, or Docker

---

### 📁 Folder Structure

