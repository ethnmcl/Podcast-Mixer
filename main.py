import os
import uuid
import subprocess
import tempfile
from typing import Optional

import requests
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, HttpUrl


# =========================
# Env
# =========================
SUPABASE_URL = os.getenv("SUPABASE_URL", "").rstrip("/")
SUPABASE_SERVICE_ROLE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY", "")
SUPABASE_BUCKET = os.getenv("SUPABASE_BUCKET", "audiofiles")
OUTPUT_PREFIX = os.getenv("OUTPUT_PREFIX", "final")  # folder inside bucket

app = FastAPI(title="Audio Mixer Service", version="1.2.0")


# =========================
# Models
# =========================
class MixRequest(BaseModel):
    voice_url: HttpUrl
    music_url: HttpUrl
    music_volume: float = Field(default=0.18, ge=0.0, le=1.0)  # 0.0 - 1.0
    duck: bool = True
    output_format: str = "mp3"
    loudnorm: bool = True


class MixPodcastRequest(BaseModel):
    voice_url: HttpUrl

    intro_music_url: HttpUrl
    outro_music_url: HttpUrl

    intro_duration: float = Field(default=8.0, ge=0.0)
    outro_duration: float = Field(default=10.0, ge=0.0)

    # smooth transitions between segments
    transition_seconds: float = Field(default=1.0, ge=0.0, le=8.0)

    # volumes
    intro_volume: float = Field(default=0.9, ge=0.0, le=1.0)
    outro_volume: float = Field(default=0.9, ge=0.0, le=1.0)

    # fades on intro/outro (seconds)
    intro_fade_in: float = Field(default=0.8, ge=0.0, le=10.0)
    intro_fade_out: float = Field(default=1.2, ge=0.0, le=10.0)
    outro_fade_in: float = Field(default=0.8, ge=0.0, le=10.0)
    outro_fade_out: float = Field(default=1.2, ge=0.0, le=10.0)

    loudnorm: bool = True


# =========================
# Helpers
# =========================
def _download_to(path: str, url: str):
    with requests.get(url, stream=True, timeout=180) as r:
        r.raise_for_status()
        with open(path, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 256):
                if chunk:
                    f.write(chunk)


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

    return f"{SUPABASE_URL}/storage/v1/object/public/{SUPABASE_BUCKET}/{object_path}"


def _run(cmd: list[str]):
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {proc.stderr[-2500:]}")


# =========================
# ffmpeg: simple mix (voice + bed)
# =========================
def _run_ffmpeg_mix(
    voice_path: str,
    music_path: str,
    out_path: str,
    music_volume: float,
    duck: bool,
    loudnorm: bool,
):
    # - Loop music to cover voice length
    # - Set music volume
    # - Optional ducking (sidechain)
    # - Mix duration = voice
    # - Optional loudnorm
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
        out_path,
    ]
    _run(cmd)


# =========================
# ffmpeg: intro + voice + outro with smooth transitions + fades
# =========================
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
    intro_fade_in: float,
    intro_fade_out: float,
    outro_fade_in: float,
    outro_fade_out: float,
    loudnorm: bool,
):
    # Normalize each input to the same format for stable crossfades
    # Then:
    # 1) trim intro/outro to durations (if >0)
    # 2) apply volume + fade in/out on intro/outro
    # 3) acrossfade intro -> voice, then voice -> outro
    # acrossfade ends up with smooth transitions without duplicating voice.

    # Safety clamps
    t = max(0.0, float(transition_seconds))

    # If user sets fade longer than duration, clamp inside duration
    if intro_duration > 0:
        intro_fade_in = min(float(intro_fade_in), float(intro_duration))
        intro_fade_out = min(float(intro_fade_out), float(intro_duration))
    if outro_duration > 0:
        outro_fade_in = min(float(outro_fade_in), float(outro_duration))
        outro_fade_out = min(float(outro_fade_out), float(outro_duration))

    filter_parts = []

    # Base format blocks
    filter_parts.append(
        "[0:a]aresample=44100,aformat=sample_fmts=fltp:channel_layouts=stereo,asetpts=N/SR/TB[voice]"
    )
    filter_parts.append(
        "[1:a]aresample=44100,aformat=sample_fmts=fltp:channel_layouts=stereo,asetpts=N/SR/TB[intro_raw]"
    )
    filter_parts.append(
        "[2:a]aresample=44100,aformat=sample_fmts=fltp:channel_layouts=stereo,asetpts=N/SR/TB[outro_raw]"
    )

    # Intro processing
    if intro_duration > 0:
        fade_out_start = max(0.0, float(intro_duration) - float(intro_fade_out))
        intro_chain = (
            f"[intro_raw]atrim=0:{intro_duration},asetpts=N/SR/TB,volume={intro_volume}"
        )
        if intro_fade_in > 0:
            intro_chain += f",afade=t=in:st=0:d={intro_fade_in}"
        if intro_fade_out > 0:
            intro_chain += f",afade=t=out:st={fade_out_start}:d={intro_fade_out}"
        intro_chain += "[intro]"
        filter_parts.append(intro_chain)
        intro_label = "[intro]"
    else:
        # If intro_duration == 0, skip intro and start from voice
        intro_label = None

    # Outro processing
    if outro_duration > 0:
        fade_out_start = max(0.0, float(outro_duration) - float(outro_fade_out))
        outro_chain = (
            f"[outro_raw]atrim=0:{outro_duration},asetpts=N/SR/TB,volume={outro_volume}"
        )
        if outro_fade_in > 0:
            outro_chain += f",afade=t=in:st=0:d={outro_fade_in}"
        if outro_fade_out > 0:
            outro_chain += f",afade=t=out:st={fade_out_start}:d={outro_fade_out}"
        outro_chain += "[outro]"
        filter_parts.append(outro_chain)
        outro_label = "[outro]"
    else:
        outro_label = None

    # Stitching logic (avoid duplicating voice)
    # Case A: intro + voice + outro (normal)
    if intro_label and outro_label:
        # acrossfade duration cannot exceed the shortest segment; ffmpeg handles but keep sane
        filter_parts.append(f"{intro_label}[voice]acrossfade=d={t}:c1=tri:c2=tri[iv]")
        filter_parts.append(f"[iv]{outro_label}acrossfade=d={t}:c1=tri:c2=tri[full]")
        final_label = "[full]"

    # Case B: intro + voice only
    elif intro_label and not outro_label:
        filter_parts.append(f"{intro_label}[voice]acrossfade=d={t}:c1=tri:c2=tri[full]")
        final_label = "[full]"

    # Case C: voice + outro only
    elif (not intro_label) and outro_label:
        filter_parts.append(f"[voice]{outro_label}acrossfade=d={t}:c1=tri:c2=tri[full]")
        final_label = "[full]"

    # Case D: just voice
    else:
        final_label = "[voice]"

    if loudnorm:
        filter_parts.append(f"{final_label}loudnorm=I=-16:TP=-1.5:LRA=11[out]")
        out_map = "[out]"
    else:
        out_map = final_label

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
        out_path,
    ]
    _run(cmd)


# =========================
# Routes
# =========================
@app.get("/")
def root():
    # Render/health checks hit "/" with GET and sometimes HEAD
    return {"ok": True, "service": "podcast-mixer"}


@app.get("/health")
def health():
    return {"ok": True}


@app.post("/mix")
def mix(req: MixRequest):
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
            "object_path": out_object,
        }

    except requests.HTTPError as e:
        raise HTTPException(status_code=400, detail=f"Download failed: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


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
                intro_fade_in=req.intro_fade_in,
                intro_fade_out=req.intro_fade_out,
                outro_fade_in=req.outro_fade_in,
                outro_fade_out=req.outro_fade_out,
                loudnorm=req.loudnorm,
            )

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

