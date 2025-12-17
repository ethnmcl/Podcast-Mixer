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

app = FastAPI(title="Audio Mixer Service", version="1.1.0")


# -----------------------------
# Models
# -----------------------------

class MixRequest(BaseModel):
    voice_url: HttpUrl
    music_url: HttpUrl
    music_volume: float = 0.18     # 0.0 - 1.0 (bed level)
    duck: bool = True             # sidechain ducking
    output_format: str = "mp3"    # "mp3" only for now
    loudnorm: bool = True         # normalize to podcast-ish loudness


class PodcastMixRequest(BaseModel):
    """
    Creates a timeline:
      [Intro music only] + [Voice + ducked music bed] + [Outro music only]

    If you want separate intro/outro tracks later, add intro_music_url/outro_music_url.
    """
    voice_url: HttpUrl
    music_url: HttpUrl

    # segment lengths
    intro_seconds: float = Field(6.0, ge=0.0)
    outro_seconds: float = Field(8.0, ge=0.0)

    # levels
    music_volume: float = Field(0.30, ge=0.0, le=1.0)   # bed during voice
    intro_volume: float = Field(0.90, ge=0.0, le=1.0)   # louder intro
    outro_volume: float = Field(0.90, ge=0.0, le=1.0)   # louder outro

    # behavior
    duck: bool = True
    loudnorm: bool = True
    output_format: str = "mp3"

    # polish
    fade_in_seconds: float = Field(0.25, ge=0.0)        # intro fade-in
    fade_out_seconds: float = Field(2.5, ge=0.0)        # outro fade-out


# -----------------------------
# Helpers
# -----------------------------

def _download_to(path: str, url: str):
    with requests.get(url, stream=True, timeout=180) as r:
        r.raise_for_status()
        with open(path, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 256):
                if chunk:
                    f.write(chunk)


def _ffmpeg_run(cmd: list[str]) -> None:
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        err = (proc.stderr or "")[-3500:]
        raise RuntimeError(f"ffmpeg failed: {err}")


def _run_ffmpeg_mix(
    voice_path: str,
    music_path: str,
    out_path: str,
    music_volume: float,
    duck: bool,
    loudnorm: bool
):
    # Loop music to cover voice length, set volume, optional ducking, mix to voice duration
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
        "ffmpeg", "-y",
        "-i", voice_path,
        "-i", music_path,
        "-filter_complex", filter_complex,
        "-map", out_map,
        "-ar", "44100",
        "-ac", "2",
        "-b:a", "192k",
        out_path
    ]
    _ffmpeg_run(cmd)


def _run_ffmpeg_podcast_timeline(
    voice_path: str,
    music_path: str,
    out_path: str,
    intro_seconds: float,
    outro_seconds: float,
    music_volume: float,
    intro_volume: float,
    outro_volume: float,
    duck: bool,
    loudnorm: bool,
    fade_in_seconds: float,
    fade_out_seconds: float,
):
    """
    Timeline:
      A) Intro: music only, trimmed to intro_seconds, optional fade in
      B) Body: voice + looped music bed (ducked), duration=voice
      C) Outro: music only, trimmed to outro_seconds, fade out

    Then concat A+B+C.
    """
    # Avoid ffmpeg complaining about fade longer than segment
    intro_fade = max(0.0, min(fade_in_seconds, intro_seconds)) if intro_seconds > 0 else 0.0
    outro_fade = max(0.0, min(fade_out_seconds, outro_seconds)) if outro_seconds > 0 else 0.0

    # Segment A: intro music only
    # Segment B: voice + bed (ducked)
    # Segment C: outro music only + fade out

    # Intro chain
    intro_chain = ""
    if intro_seconds > 0:
        intro_chain = (
            f"[1:a]atrim=0:{intro_seconds},asetpts=PTS-STARTPTS,"
            f"volume={intro_volume}"
        )
        if intro_fade > 0:
            intro_chain += f",afade=t=in:st=0:d={intro_fade}"
        intro_chain += "[intro];"
    else:
        # generate silent 0-length segment not needed; we will skip concat inputs later
        pass

    # Body chain (voice + bed)
    if duck:
        body_chain = (
            f"[1:a]aloop=loop=-1:size=2147483647,volume={music_volume}[m];"
            f"[m][0:a]sidechaincompress=threshold=0.02:ratio=8:attack=5:release=2000[bed];"
            f"[0:a][bed]amix=inputs=2:duration=first:dropout_transition=3,asetpts=PTS-STARTPTS[body];"
        )
    else:
        body_chain = (
            f"[1:a]aloop=loop=-1:size=2147483647,volume={music_volume}[m];"
            f"[0:a][m]amix=inputs=2:duration=first:dropout_transition=3,asetpts=PTS-STARTPTS[body];"
        )

    # Outro chain
    outro_chain = ""
    if outro_seconds > 0:
        # fade-out starts at (outro_seconds - outro_fade)
        fade_start = max(0.0, outro_seconds - outro_fade) if outro_fade > 0 else 0.0
        outro_chain = (
            f"[1:a]atrim=0:{outro_seconds},asetpts=PTS-STARTPTS,"
            f"volume={outro_volume}"
        )
        if outro_fade > 0:
            outro_chain += f",afade=t=out:st={fade_start}:d={outro_fade}"
        outro_chain += "[outro];"
    else:
        pass

    # Concat inputs list (only include segments that exist)
    segs = []
    if intro_seconds > 0:
        segs.append("[intro]")
    segs.append("[body]")
    if outro_seconds > 0:
        segs.append("[outro]")

    concat_n = len(segs)
    concat_chain = "".join(segs) + f"concat=n={concat_n}:v=0:a=1[cat]"

    filter_complex = (intro_chain + body_chain + outro_chain + concat_chain)

    if loudnorm:
        filter_complex += ";[cat]loudnorm=I=-16:TP=-1.5:LRA=11[out]"
        out_map = "[out]"
    else:
        out_map = "[cat]"

    cmd = [
        "ffmpeg", "-y",
        "-i", voice_path,
        "-i", music_path,
        "-filter_complex", filter_complex,
        "-map", out_map,
        "-ar", "44100",
        "-ac", "2",
        "-b:a", "192k",
        out_path
    ]
    _ffmpeg_run(cmd)


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
        raise RuntimeError(f"Supabase upload failed: {r.status_code} {r.text[:800]}")

    public_url = f"{SUPABASE_URL}/storage/v1/object/public/{SUPABASE_BUCKET}/{object_path}"
    return public_url


# -----------------------------
# Routes
# -----------------------------

@app.get("/")
def root():
    # Render health checks / browser checks tend to hit /
    return {"ok": True, "service": "audio-mixer", "version": "1.1.0"}


@app.head("/")
def root_head():
    # Some infra uses HEAD /
    return {}


@app.get("/health")
def health():
    return {"ok": True}


@app.post("/mix")
def mix(req: MixRequest):
    if req.output_format.lower() != "mp3":
        raise HTTPException(status_code=400, detail="Only output_format=mp3 is supported right now")
    if not (0.0 <= req.music_volume <= 1.0):
        raise HTTPException(status_code=400, detail="music_volume must be between 0.0 and 1.0")

    job_id = str(uuid.uuid4())
    out_object = f"{OUTPUT_PREFIX}/{job_id}.mp3"

    try:
        with tempfile.TemporaryDirectory() as td:
            voice_path = os.path.join(td, "voice")
            music_path = os.path.join(td, "music")
            out_path = os.path.join(td, "final.mp3")

            _download_to(voice_path, str(req.voice_url))
            _download_to(music_path, str(req.music_url))

            _run_ffmpeg_mix(
                voice_path=voice_path,
                music_path=music_path,
                out_path=out_path,
                music_volume=req.music_volume,
                duck=req.duck,
                loudnorm=req.loudnorm,
            )

            final_url = _supabase_upload(out_path, out_object, content_type="audio/mpeg")

        return {
            "ok": True,
            "job_id": job_id,
            "final_url": final_url,
            "bucket": SUPABASE_BUCKET,
            "object_path": out_object
        }

    except requests.HTTPError as e:
        raise HTTPException(status_code=400, detail=f"Download failed: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/mix_podcast")
def mix_podcast(req: PodcastMixRequest):
    if req.output_format.lower() != "mp3":
        raise HTTPException(status_code=400, detail="Only output_format=mp3 is supported right now")

    job_id = str(uuid.uuid4())
    out_object = f"{OUTPUT_PREFIX}/{job_id}.mp3"

    try:
        with tempfile.TemporaryDirectory() as td:
            voice_path = os.path.join(td, "voice")
            music_path = os.path.join(td, "music")
            out_path = os.path.join(td, "final.mp3")

            _download_to(voice_path, str(req.voice_url))
            _download_to(music_path, str(req.music_url))

            _run_ffmpeg_podcast_timeline(
                voice_path=voice_path,
                music_path=music_path,
                out_path=out_path,
                intro_seconds=req.intro_seconds,
                outro_seconds=req.outro_seconds,
                music_volume=req.music_volume,
                intro_volume=req.intro_volume,
                outro_volume=req.outro_volume,
                duck=req.duck,
                loudnorm=req.loudnorm,
                fade_in_seconds=req.fade_in_seconds,
                fade_out_seconds=req.fade_out_seconds,
            )

            final_url = _supabase_upload(out_path, out_object, content_type="audio/mpeg")

        return {
            "ok": True,
            "job_id": job_id,
            "final_url": final_url,
            "bucket": SUPABASE_BUCKET,
            "object_path": out_object,
            "segments": {
                "intro_seconds": req.intro_seconds,
                "outro_seconds": req.outro_seconds
            }
        }

    except requests.HTTPError as e:
        raise HTTPException(status_code=400, detail=f"Download failed: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
