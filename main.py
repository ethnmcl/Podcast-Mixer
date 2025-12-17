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
OUTPUT_PREFIX = os.getenv("OUTPUT_PREFIX", "final")  # folder inside bucket

app = FastAPI(title="Audio Mixer Service", version="1.2.0")


# -----------------------------
# Models
# -----------------------------

class MixRequest(BaseModel):
    voice_url: HttpUrl
    music_url: HttpUrl
    music_volume: float = 0.18     # 0.0 - 1.0
    duck: bool = True             # sidechain ducking
    output_format: str = "mp3"    # "mp3" only for now
    loudnorm: bool = True         # normalize to podcast-ish loudness


class PodcastMixRequest(BaseModel):
    voice_url: HttpUrl
    bed_url: HttpUrl                 # background music under voice (looped)
    intro_url: Optional[HttpUrl] = None
    outro_url: Optional[HttpUrl] = None

    bed_volume: float = 0.18         # 0.0 - 1.0
    duck: bool = True                # duck bed under voice
    loudnorm: bool = True            # normalize to podcast-ish loudness

    # Smooth transitions + carryover
    intro_xfade_sec: float = 1.50      # crossfade intro -> podcast
    intro_under_voice_sec: float = 4.0 # intro continues quietly under voice for N seconds
    intro_under_voice_volume: float = 0.10  # volume of intro during carryover (0-1)

    outro_xfade_sec: float = 1.50      # crossfade podcast -> outro

    # Duration controls (cap intro/outro length)
    intro_max_sec: Optional[float] = 12.0   # only take first X seconds of intro (None = full)
    outro_max_sec: Optional[float] = 12.0   # only take first X seconds of outro (None = full)

    output_format: str = "mp3"          # mp3 only


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


def _ffprobe_duration(path: str) -> float:
    cmd = [
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        path
    ]
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode != 0:
        raise RuntimeError(f"ffprobe failed: {p.stderr[-2000:]}")
    try:
        return float(p.stdout.strip())
    except Exception:
        raise RuntimeError(f"ffprobe returned invalid duration: {p.stdout!r}")


def _run(cmd: list[str]):
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {proc.stderr[-2000:]}")


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
        r = requests.post(upload_url, headers=headers, data=f, timeout=180)

    if r.status_code not in (200, 201):
        raise RuntimeError(f"Supabase upload failed: {r.status_code} {r.text[:500]}")

    public_url = f"{SUPABASE_URL}/storage/v1/object/public/{SUPABASE_BUCKET}/{object_path}"
    return public_url


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


# -----------------------------
# Core audio ops
# -----------------------------

def _run_ffmpeg_mix(voice_path: str, music_path: str, out_path: str, music_volume: float, duck: bool, loudnorm: bool):
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
    _run(cmd)


def _run_ffmpeg_podcast(
    voice_path: str,
    bed_path: str,
    out_path: str,
    bed_volume: float,
    duck: bool,
    loudnorm: bool,
    intro_path: Optional[str],
    outro_path: Optional[str],
    intro_xfade_sec: float,
    intro_under_voice_sec: float,
    intro_under_voice_volume: float,
    outro_xfade_sec: float,
    intro_max_sec: Optional[float],
    outro_max_sec: Optional[float],
):
    bed_volume = _clamp01(bed_volume)
    intro_under_voice_volume = _clamp01(intro_under_voice_volume)

    voice_dur = _ffprobe_duration(voice_path)

    intro_dur = _ffprobe_duration(intro_path) if intro_path else 0.0
    outro_dur = _ffprobe_duration(outro_path) if outro_path else 0.0

    # Apply caps
    if intro_path and intro_max_sec is not None:
        intro_dur = min(intro_dur, float(intro_max_sec))
    if outro_path and outro_max_sec is not None:
        outro_dur = min(outro_dur, float(outro_max_sec))

    def clamp_xfade(xfade: float, a: float, b: float) -> float:
        # acrossfade duration must be < both segments
        if a <= 0 or b <= 0:
            return 0.0
        m = max(0.0, min(float(xfade), a - 0.05, b - 0.05))
        return max(0.0, m)

    # Build base podcast mix [pod] = voice + looped bed (duck optional)
    if duck:
        pod_mix = (
            f"[bed]aloop=loop=-1:size=2147483647,volume={bed_volume}[m];"
            f"[m][voice]sidechaincompress=threshold=0.02:ratio=8:attack=5:release=2000[bg];"
            f"[voice][bg]amix=inputs=2:duration=first:dropout_transition=3[pod]"
        )
    else:
        pod_mix = (
            f"[bed]aloop=loop=-1:size=2147483647,volume={bed_volume}[m];"
            f"[voice][m]amix=inputs=2:duration=first:dropout_transition=3[pod]"
        )

    # Inputs ordering:
    # 0 voice, 1 bed, 2 intro (optional), 3 outro (optional)
    parts = [
        "[0:a]asetpts=PTS-STARTPTS[voice]",
        "[1:a]asetpts=PTS-STARTPTS[bed]",
        pod_mix
    ]

    current = "[pod]"

    # -----------------------------
    # Intro with carryover:
    # Make a short "pod_start" segment of length intro_under_voice_sec from the podcast mix,
    # then mix that with the last intro_under_voice_sec of the intro (at low volume + fade out).
    # Then we crossfade the full intro into that "pod_start_with_intro" segment.
    # Finally, append the remainder of the podcast mix after that start segment.
    # -----------------------------
    if intro_path:
        # Trim intro to max length
        intro_trim = f"[2:a]atrim=start=0:end={intro_dur},asetpts=PTS-STARTPTS[intro]"
        parts.append(intro_trim)

        # Ensure under-voice sec is sane
        under = max(0.0, float(intro_under_voice_sec))
        under = min(under, max(0.0, voice_dur - 0.05))
        under = min(under, max(0.0, intro_dur - 0.05)) if intro_dur > 0 else 0.0

        if under > 0:
            # Take first "under" seconds of podcast mix
            parts.append(f"{current}atrim=start=0:end={under},asetpts=PTS-STARTPTS[pod_start]")
            # Take last "under" seconds of intro (end portion), lower volume, fade out over 'under'
            intro_tail_start = max(0.0, intro_dur - under)
            parts.append(
                f"[intro]atrim=start={intro_tail_start}:end={intro_dur},asetpts=PTS-STARTPTS[intro_tail];"
                f"[intro_tail]volume={intro_under_voice_volume},afade=t=out:st=0:d={under}[intro_tail_q]"
            )
            # Mix intro tail under the podcast start (carryover)
            parts.append("[pod_start][intro_tail_q]amix=inputs=2:duration=first:dropout_transition=2[pod_start_carry]")
            pod_start_seg = "[pod_start_carry]"
        else:
            # No carryover requested
            parts.append(f"{current}atrim=start=0:end=2,asetpts=PTS-STARTPTS[pod_start_fallback]")
            pod_start_seg = "[pod_start_fallback]"

        # Remaining podcast after the start segment
        parts.append(f"{current}atrim=start={under},asetpts=PTS-STARTPTS[pod_rest]")

        # Crossfade full intro into (pod_start + carryover)
        # crossfade duration must be < intro AND < pod_start length
        intro_x = clamp_xfade(float(intro_xfade_sec), intro_dur, under if under > 0 else 2.0)
        if intro_x > 0:
            parts.append(f"[intro]{pod_start_seg}acrossfade=d={intro_x}:curve1=tri:curve2=tri[pod_with_intro]")
        else:
            parts.append(f"[intro]{pod_start_seg}concat=n=2:v=0:a=1[pod_with_intro]")

        # Stitch: (intro->pod_start_with_carry) + pod_rest
        parts.append("[pod_with_intro][pod_rest]concat=n=2:v=0:a=1[pod_after_intro]")
        current = "[pod_after_intro]"

    # -----------------------------
    # Outro crossfade (with cap)
    # -----------------------------
    if outro_path:
        parts.append(f"[3:a]atrim=start=0:end={outro_dur},asetpts=PTS-STARTPTS[outro]")

        outro_x = clamp_xfade(float(outro_xfade_sec), _ffprobe_duration(voice_path) + 60.0, outro_dur)
        # safer: clamp against current segment length by just ensuring outro_x <= outro_dur; acrossfade also needs current >= outro_x.
        # In practice current will be long; but to be safe, also limit to 6s.
        outro_x = max(0.0, min(outro_x, 6.0, outro_dur - 0.05 if outro_dur > 0 else 0.0))

        if outro_x > 0:
            parts.append(f"{current}[outro]acrossfade=d={outro_x}:curve1=tri:curve2=tri[final]")
            current = "[final]"
        else:
            parts.append(f"{current}[outro]concat=n=2:v=0:a=1[final]")
            current = "[final]"

    # Final normalize at the end
    if loudnorm:
        parts.append(f"{current}loudnorm=I=-16:TP=-1.5:LRA=11[out]")
        out_map = "[out]"
    else:
        out_map = current

    filter_complex = ";".join(parts)

    cmd = ["ffmpeg", "-y", "-i", voice_path, "-i", bed_path]
    if intro_path:
        cmd += ["-i", intro_path]
    if outro_path:
        cmd += ["-i", outro_path]

    cmd += [
        "-filter_complex", filter_complex,
        "-map", out_map,
        "-ar", "44100",
        "-ac", "2",
        "-b:a", "192k",
        out_path
    ]

    _run(cmd)


# -----------------------------
# Routes
# -----------------------------

@app.get("/")
def root():
    return {"ok": True, "service": "podcast-mixer", "version": "1.2.0"}


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
                loudnorm=req.loudnorm
            )

            final_url = _supabase_upload(out_path, out_object, content_type="audio/mpeg")

        return {"ok": True, "job_id": job_id, "final_url": final_url, "bucket": SUPABASE_BUCKET, "object_path": out_object}

    except requests.HTTPError as e:
        raise HTTPException(status_code=400, detail=f"Download failed: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/mix_podcast")
def mix_podcast(req: PodcastMixRequest):
    if req.output_format.lower() != "mp3":
        raise HTTPException(status_code=400, detail="Only output_format=mp3 is supported right now")
    if not (0.0 <= req.bed_volume <= 1.0):
        raise HTTPException(status_code=400, detail="bed_volume must be between 0.0 and 1.0")
    if req.intro_xfade_sec < 0 or req.outro_xfade_sec < 0:
        raise HTTPException(status_code=400, detail="intro_xfade_sec/outro_xfade_sec must be >= 0")
    if req.intro_under_voice_sec < 0:
        raise HTTPException(status_code=400, detail="intro_under_voice_sec must be >= 0")
    if req.intro_under_voice_volume < 0 or req.intro_under_voice_volume > 1:
        raise HTTPException(status_code=400, detail="intro_under_voice_volume must be between 0 and 1")
    if req.intro_max_sec is not None and req.intro_max_sec <= 0:
        raise HTTPException(status_code=400, detail="intro_max_sec must be > 0 or null")
    if req.outro_max_sec is not None and req.outro_max_sec <= 0:
        raise HTTPException(status_code=400, detail="outro_max_sec must be > 0 or null")

    job_id = str(uuid.uuid4())
    out_object = f"{OUTPUT_PREFIX}/{job_id}.mp3"

    try:
        with tempfile.TemporaryDirectory() as td:
            voice_path = os.path.join(td, "voice")
            bed_path = os.path.join(td, "bed")
            intro_path = os.path.join(td, "intro") if req.intro_url else None
            outro_path = os.path.join(td, "outro") if req.outro_url else None
            out_path = os.path.join(td, "final.mp3")

            _download_to(voice_path, str(req.voice_url))
            _download_to(bed_path, str(req.bed_url))
            if req.intro_url and intro_path:
                _download_to(intro_path, str(req.intro_url))
            if req.outro_url and outro_path:
                _download_to(outro_path, str(req.outro_url))

            _run_ffmpeg_podcast(
                voice_path=voice_path,
                bed_path=bed_path,
                out_path=out_path,
                bed_volume=req.bed_volume,
                duck=req.duck,
                loudnorm=req.loudnorm,
                intro_path=intro_path,
                outro_path=outro_path,
                intro_xfade_sec=req.intro_xfade_sec,
                intro_under_voice_sec=req.intro_under_voice_sec,
                intro_under_voice_volume=req.intro_under_voice_volume,
                outro_xfade_sec=req.outro_xfade_sec,
                intro_max_sec=req.intro_max_sec,
                outro_max_sec=req.outro_max_sec,
            )

            final_url = _supabase_upload(out_path, out_object, content_type="audio/mpeg")

        return {"ok": True, "job_id": job_id, "final_url": final_url, "bucket": SUPABASE_BUCKET, "object_path": out_object}

    except requests.HTTPError as e:
        raise HTTPException(status_code=400, detail=f"Download failed: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

