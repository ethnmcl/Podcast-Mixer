import os
import uuid
import subprocess
import tempfile
from typing import Optional

import requests
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, HttpUrl

SUPABASE_URL = os.getenv("SUPABASE_URL", "").rstrip("/")
SUPABASE_SERVICE_ROLE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY", "")
SUPABASE_BUCKET = os.getenv("SUPABASE_BUCKET", "audiofiles")
OUTPUT_PREFIX = os.getenv("OUTPUT_PREFIX", "final")

app = FastAPI(title="Podcast Mixer Service", version="2.0.0")


class MixPodcastRequest(BaseModel):
    voice_url: HttpUrl
    intro_music_url: HttpUrl
    outro_music_url: HttpUrl

    intro_duration: float = 8.0   # seconds
    outro_duration: float = 10.0  # seconds

    intro_volume: float = 1.0     # 0.0 - 2.0
    outro_volume: float = 1.0     # 0.0 - 2.0

    fade_in: float = 0.8          # seconds
    fade_out: float = 1.2         # seconds

    loudnorm: bool = True
    output_format: str = "mp3"


def _download_to(path: str, url: str):
    with requests.get(url, stream=True, timeout=180) as r:
        r.raise_for_status()
        with open(path, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 256):
                if chunk:
                    f.write(chunk)


def _run(cmd: list[str]):
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {proc.stderr[-2000:]}")
    return proc


def _supabase_upload(file_path: str, object_path: str, content_type: str = "audio/mpeg") -> str:
    if not SUPABASE_URL or not SUPABASE_SERVICE_ROLE_KEY:
        raise RuntimeError("Missing SUPABASE_URL or SUPABASE_SERVICE_ROLE_KEY env vars")

    upload_url = f"{SUPABASE_URL}/storage/v1/object/{SUPABASE_BUCKET}/{object_path}"
    headers = {
        "Authorization": f"Bearer {SUPABASE_SERVICE_ROLE_KEY}",
        "apikey": SUPABASE_SERVICE_ROLE_KEY,
        "Content-Type": content_type,
        "x-upsert": "true",
    }

    with open(file_path, "rb") as f:
        r = requests.post(upload_url, headers=headers, data=f, timeout=240)

    if r.status_code not in (200, 201):
        raise RuntimeError(f"Supabase upload failed: {r.status_code} {r.text[:500]}")

    return f"{SUPABASE_URL}/storage/v1/object/public/{SUPABASE_BUCKET}/{object_path}"


@app.get("/health")
def health():
    return {"ok": True}


@app.post("/mix_podcast")
def mix_podcast(req: MixPodcastRequest):
    if req.output_format.lower() != "mp3":
        raise HTTPException(status_code=400, detail="Only output_format=mp3 is supported right now")

    if req.intro_duration <= 0 or req.outro_duration <= 0:
        raise HTTPException(status_code=400, detail="intro_duration and outro_duration must be > 0")

    if not (0.0 <= req.intro_volume <= 2.0) or not (0.0 <= req.outro_volume <= 2.0):
        raise HTTPException(status_code=400, detail="intro_volume/outro_volume must be between 0.0 and 2.0")

    if req.fade_in < 0 or req.fade_out < 0:
        raise HTTPException(status_code=400, detail="fade_in/fade_out must be >= 0")

    job_id = str(uuid.uuid4())
    out_object = f"{OUTPUT_PREFIX}/{job_id}.mp3"

    try:
        with tempfile.TemporaryDirectory() as td:
            voice_path = os.path.join(td, "voice.mp3")
            intro_path = os.path.join(td, "intro.mp3")
            outro_path = os.path.join(td, "outro.mp3")
            out_path = os.path.join(td, "final.mp3")

            _download_to(voice_path, str(req.voice_url))
            _download_to(intro_path, str(req.intro_music_url))
            _download_to(outro_path, str(req.outro_music_url))

            # Build: intro(trim+fade) -> voice -> outro(trim+fade) using concat
            # Ensure fade durations don't exceed clip durations
            intro_fade_out = min(req.fade_out, max(0.0, req.intro_duration - 0.001))
            outro_fade_out = min(req.fade_out, max(0.0, req.outro_duration - 0.001))

            filter_complex = (
                # Intro
                f"[1:a]atrim=0:{req.intro_duration},"
                f"afade=t=in:st=0:d={req.fade_in},"
                f"afade=t=out:st={max(0.0, req.intro_duration - intro_fade_out)}:d={intro_fade_out},"
                f"volume={req.intro_volume},"
                f"asetpts=N/SR/TB[intro];"
                # Voice (normalize timestamps)
                f"[0:a]asetpts=N/SR/TB[voice];"
                # Outro
                f"[2:a]atrim=0:{req.outro_duration},"
                f"afade=t=in:st=0:d={req.fade_in},"
                f"afade=t=out:st={max(0.0, req.outro_duration - outro_fade_out)}:d={outro_fade_out},"
                f"volume={req.outro_volume},"
                f"asetpts=N/SR/TB[outro];"
                # Concat segments
                f"[intro][voice][outro]concat=n=3:v=0:a=1[cat]"
            )

            if req.loudnorm:
                filter_complex += ";[cat]loudnorm=I=-16:TP=-1.5:LRA=11[out]"
                out_map = "[out]"
            else:
                out_map = "[cat]"

            cmd = [
                "ffmpeg", "-y",
                "-i", voice_path,
                "-i", intro_path,
                "-i", outro_path,
                "-filter_complex", filter_complex,
                "-map", out_map,
                "-ar", "44100",
                "-ac", "2",
                "-b:a", "192k",
                out_path,
            ]
            _run(cmd)

            final_url = _supabase_upload(out_path, out_object, content_type="audio/mpeg")

        return {
            "ok": True,
            "job_id": job_id,
            "final_url": final_url,
            "bucket": SUPABASE_BUCKET,
            "object_path": out_object,
        }

    except requests.HTTPError as e:
        raise HTTPException(status_code=400, detail=f"Download failed: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
