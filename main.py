import os
import uuid
import subprocess
import tempfile

import requests
from fastapi import FastAPI, HTTPException, Response
from pydantic import BaseModel, HttpUrl


# -----------------------------
# Environment
# -----------------------------
SUPABASE_URL = os.getenv("SUPABASE_URL", "").rstrip("/")
SUPABASE_SERVICE_ROLE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY", "")
SUPABASE_BUCKET = os.getenv("SUPABASE_BUCKET", "audiofiles")
OUTPUT_PREFIX = os.getenv("OUTPUT_PREFIX", "final")


# -----------------------------
# App
# -----------------------------
app = FastAPI(title="Audio Mixer Service", version="1.0.0")


# -----------------------------
# Models
# -----------------------------
class MixRequest(BaseModel):
    voice_url: HttpUrl
    music_url: HttpUrl
    music_volume: float = 0.18
    duck: bool = True
    output_format: str = "mp3"
    loudnorm: bool = True


# -----------------------------
# Helpers
# -----------------------------
def _download_to(path: str, url: str):
    with requests.get(url, stream=True, timeout=120) as r:
        r.raise_for_status()
        with open(path, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 256):
                if chunk:
                    f.write(chunk)


def _run_ffmpeg_mix(
    voice_path: str,
    music_path: str,
    out_path: str,
    music_volume: float,
    duck: bool,
    loudnorm: bool,
):
    if duck:
        filter_complex = (
            f"[1:a]aloop=loop=-1:size=2147483647,volume={music_volume}[m];"
            f"[m][0:a]sidechaincompress=threshold=0.02:ratio=8:attack=5:release=2000[bg];"
            f"[0:a][bg]amix=inputs=2:duration=first:dropout_transition=3[mix]"
        )
    else:
        filter_complex = (
            f"[1:a]aloop=loop=-1:size=2147483647,volume={music_volume}[m];"
            f"[0:a][m]amix=inputs=2:duration=first:dropout_transition=3[mix]"
        )

    if loudnorm:
        filter_complex += ";[mix]loudnorm=I=-16:TP=-1.5:LRA=11[out]"
        out_map = "[out]"
    else:
        out_map = "[mix]"

    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        voice_path,
        "-i",
        music_path,
        "-filter_complex",
        filter_complex,
        "-map",
        out_map,
        "-ar",
        "44100",
        "-ac",
        "2",
        "-b:a",
        "192k",
        out_path,
    ]

    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {proc.stderr[-2000:]}")


def _supabase_upload(file_path: str, object_path: str) -> str:
    if not SUPABASE_URL or not SUPABASE_SERVICE_ROLE_KEY:
        raise RuntimeError("Missing Supabase credentials")

    upload_url = f"{SUPABASE_URL}/storage/v1/object/{SUPABASE_BUCKET}/{object_path}"
    headers = {
        "Authorization": f"Bearer {SUPABASE_SERVICE_ROLE_KEY}",
        "apikey": SUPABASE_SERVICE_ROLE_KEY,
        "Content-Type": "audio/mpeg",
        "x-upsert": "true",
    }

    with open(file_path, "rb") as f:
        r = requests.post(upload_url, headers=headers, data=f, timeout=180)

    if r.status_code not in (200, 201):
        raise RuntimeError(f"Supabase upload failed: {r.status_code} {r.text[:500]}")

    return f"{SUPABASE_URL}/storage/v1/object/public/{SUPABASE_BUCKET}/{object_path}"


# -----------------------------
# Root + Health (Render-safe)
# -----------------------------
@app.head("/")
def root_head():
    return Response(status_code=200)


@app.get("/")
def root_get():
    return {
        "ok": True,
        "service": "audio-mixer",
        "health": "/health",
        "mix": "/mix",
        "docs": "/docs",
    }


@app.get("/health")
def health():
    return {"ok": True}


# -----------------------------
# Main API
# -----------------------------
@app.post("/mix")
def mix(req: MixRequest):
    if req.output_format.lower() != "mp3":
        raise HTTPException(status_code=400, detail="Only mp3 output is supported")

    if not (0.0 <= req.music_volume <= 1.0):
        raise HTTPException(status_code=400, detail="music_volume must be 0.0–1.0")

    job_id = str(uuid.uuid4())
    object_path = f"{OUTPUT_PREFIX}/{job_id}.mp3"

    try:
        with tempfile.TemporaryDirectory() as td:
            voice_path = os.path.join(td, "voice")
            music_path = os.path.join(td, "music")
            out_path = os.path.join(td, "final.mp3")

            _download_to(voice_path, str(req.voice_url))
            _download_to(music_path, str(req.music_url))

            _run_ffmpeg_mix(
                voice_path,
                music_path,
                out_path,
                req.music_volume,
                req.duck,
                req.loudnorm,
            )

            final_url = _supabase_upload(out_path, object_path)

        return {
            "ok": True,
            "job_id": job_id,
            "final_url": final_url,
            "bucket": SUPABASE_BUCKET,
            "object_path": object_path,
        }

    except requests.HTTPError as e:
        raise HTTPException(status_code=400, detail=f"Download failed: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
