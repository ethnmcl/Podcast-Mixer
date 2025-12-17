import os
import uuid
import subprocess
import tempfile
from typing import Optional

import requests
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, HttpUrl, Field


SUPABASE_URL = os.getenv("SUPABASE_URL", "").rstrip("/")
SUPABASE_SERVICE_ROLE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY", "")
SUPABASE_BUCKET = os.getenv("SUPABASE_BUCKET", "audiofiles")
OUTPUT_PREFIX = os.getenv("OUTPUT_PREFIX", "final")  # folder inside bucket

app = FastAPI(title="Podcast Mixer Service", version="1.1.0")


# ---------- Request Models ----------

class MixPodcastRequest(BaseModel):
    voice_url: HttpUrl

    intro_music_url: HttpUrl
    outro_music_url: HttpUrl

    # Hard trim lengths (seconds)
    intro_duration: float = Field(default=8.0, ge=0.0)
    outro_duration: float = Field(default=10.0, ge=0.0)

    # Smooth transitions (crossfade seconds)
    transition_seconds: float = Field(default=1.0, ge=0.0, le=8.0)

    # Music volumes
    intro_volume: float = Field(default=0.9, ge=0.0, le=1.0)
    outro_volume: float = Field(default=0.9, ge=0.0, le=1.0)

    # Optional mastering
    loudnorm: bool = True


# ---------- Helpers ----------

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
        raise RuntimeError(proc.stderr[-2500:])


def _run_ffmpeg_podcast(
    voice_path: str,
    intro_path: str,
    outro_path: str,
    out_path: str,
    intro_duration: float,
    outro_duration: float,
    transition_seconds: float,
    intro_volume: float,
    outro_volume: float,
    loudnorm: bool,
):
    # If user sets transition bigger than the trim duration, FFmpeg can error.
    # Keep it safe:
    if intro_duration > 0 and transition_seconds >= intro_duration:
        transition_seconds = max(0.0, intro_duration - 0.05)
    if outro_duration > 0 and transition_seconds >= outro_duration:
        transition_seconds = max(0.0, outro_duration - 0.05)

    # Build filtergraph:
    # - Trim intro/outro to duration
    # - Apply volume
    # - Apply fades at ends (so crossfade isn’t harsh)
    # - Crossfade intro->voice, then ->outro
    #
    # acrossfade notes:
    # - d=transition_seconds sets crossfade length
    # - c1/c2 choose curve; "tri" is smooth and common for podcasts
    #
    # We also reset timestamps with asetpts to keep concat/crossfade stable.
    filter_parts = []

    # Intro
    if intro_duration > 0:
        # fade out intro for the last transition_seconds
        fade_out_start = max(0.0, intro_duration - transition_seconds) if transition_seconds > 0 else intro_duration
        intro_chain = (
            f"[1:a]atrim=0:{intro_duration},asetpts=N/SR/TB,"
            f"volume={intro_volume}"
        )
        if transition_seconds > 0:
            intro_chain += f",afade=t=out:st={fade_out_start}:d={transition_seconds}"
        intro_chain += "[intro]"
        filter_parts.append(intro_chain)
        intro_label = "[intro]"
    else:
        intro_label = None

    # Voice
    voice_chain = "[0:a]asetpts=N/SR/TB[voice]"
    filter_parts.append(voice_chain)

    # Outro
    if outro_duration > 0:
        outro_chain = (
            f"[2:a]atrim=0:{outro_duration},asetpts=N/SR/TB,"
            f"volume={outro_volume}"
        )
        if transition_seconds > 0:
            outro_chain += f",afade=t=in:st=0:d={transition_seconds}"
        outro_chain += "[outro]"
        filter_parts.append(outro_chain)
        outro_label = "[outro]"
    else:
        outro_label = None

    # Crossfades
    if intro_label and transition_seconds > 0:
        filter_parts.append(
            f"{intro_label}[voice]acrossfade=d={transition_seconds}:c1=tri:c2=tri[m1]"
        )
        mid = "[m1]"
    elif intro_label:
        # no crossfade: concat intro + voice
        filter_parts.append(f"{intro_label}[voice]concat=n=2:v=0:a=1[m1]")
        mid = "[m1]"
    else:
        mid = "[voice]"

    if outro_label and transition_seconds > 0:
        filter_parts.append(
            f"{mid}{outro_label}acrossfade=d={transition_seconds}:c1=tri:c2=tri[mix]"
        )
        final = "[mix]"
    elif outro_label:
        filter_parts.append(f"{mid}{outro_label}concat=n=2:v=0:a=1[mix]")
        final = "[mix]"
    else:
        final = mid

    if loudnorm:
        filter_parts.append(f"{final}loudnorm=I=-16:TP=-1.5:LRA=11[out]")
        out_map = "[out]"
    else:
        out_map = final

    filter_complex = ";".join(filter_parts)

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
        out_path
    ]
    _run(cmd)


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

    public_url = f"{SUPABASE_URL}/storage/v1/object/public/{SUPABASE_BUCKET}/{object_path}"
    return public_url


# ---------- Routes ----------

@app.get("/health")
def health():
    return {"ok": True}


@app.post("/mix_podcast")
def mix_podcast(req: MixPodcastRequest):
    job_id = str(uuid.uuid4())
    out_object = f"{OUTPUT_PREFIX}/{job_id}.mp3"

    try:
        with tempfile.TemporaryDirectory() as td:
            voice_path = os.path.join(td, "voice")
            intro_path = os.path.join(td, "intro")
            outro_path = os.path.join(td, "outro")
            out_path = os.path.join(td, "final.mp3")

            _download_to(voice_path, str(req.voice_url))
            _download_to(intro_path, str(req.intro_music_url))
            _download_to(outro_path, str(req.outro_music_url))

            _run_ffmpeg_podcast(
                voice_path=voice_path,
                intro_path=intro_path,
                outro_path=outro_path,
                out_path=out_path,
                intro_duration=req.intro_duration,
                outro_duration=req.outro_duration,
                transition_seconds=req.transition_seconds,
                intro_volume=req.intro_volume,
                outro_volume=req.outro_volume,
                loudnorm=req.loudnorm,
            )

            final_url = _supabase_upload(out_path, out_object, content_type="audio/mpeg")

        return {
            "ok": True,
            "job_id": job_id,
            "final_url": final_url,
            "bucket": SUPABASE_BUCKET,
            "object_path": out_object,
            "transition_seconds": req.transition_seconds,
            "intro_duration": req.intro_duration,
            "outro_duration": req.outro_duration,
        }

    except requests.HTTPError as e:
        raise HTTPException(status_code=400, detail=f"Download failed: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
