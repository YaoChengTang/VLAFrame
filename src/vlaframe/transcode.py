"""Function 1 — bring a video into the format the reader is fast on.

Two knobs matter for random single-frame access, and they are not the ones people
usually reach for:

``gop``       how far the decoder must walk from a keyframe to reach an arbitrary
              frame. Cost is linear in that distance, so this is the dominant
              latency knob -- and the dominant storage knob, in the opposite
              direction. Measured at 640x480: GOP 30 costs 6.07 MB/1500 frames
              and ~10 ms/frame; GOP 8 costs 9.21 MB and ~4 ms.

``bframes``   B-frames that nothing references are skipped outright by the reader
              (see index.py), so they are nearly free to decode past. But they
              also introduce reorder delay, which is what every previous
              frame-indexing bug in this corpus turned out to be. ``bframes=0``
              removes the reorder entirely at about +5% storage.

``crf`` and ``profile`` change quality and compatibility, not seek behaviour.

Correctness
-----------
The output is frame-count-verified against the source before it replaces
anything: the h5/action arrays index these frames positionally, so a transcode
that drops or adds one silently misaligns every downstream timestep. Encoding
happens in a scratch directory and the result is moved into place, so an
interrupted run never leaves a half-written file where a complete one is
expected.
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

_STREAM_RE = re.compile(r"Stream #\d+:\d+.*?: Video: ([A-Za-z0-9_]+).*?, (\d+)x(\d+)")
_FRAME_RE = re.compile(r"frame=\s*(\d+)")


def ffmpeg_bin() -> str:
    """Prefer an explicitly configured binary, then the pip-installable static
    build, then whatever is on PATH. Cloud images routinely ship no ffmpeg and
    no root to install one, which is why imageio-ffmpeg is in the middle."""
    if os.environ.get("FFMPEG_BIN"):
        return os.environ["FFMPEG_BIN"]
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        found = shutil.which("ffmpeg")
        if not found:
            raise RuntimeError(
                "no ffmpeg found: set FFMPEG_BIN, or `pip install imageio-ffmpeg`")
        return found


@dataclass
class TranscodeSpec:
    """The target format. Defaults are what this package's benchmarks were taken
    on, so they are a known-good starting point rather than a recommendation for
    every corpus."""
    gop: int = 30
    crf: int = 23
    bframes: int | None = None      # None = leave x264's default (3, with pyramid)
    profile: str = "high"
    pix_fmt: str = "yuv420p"
    preset: str = "medium"
    max_side: int | None = None     # cap the LONGER edge, preserving aspect
    codec: str = "libx264"
    extra_args: list[str] = field(default_factory=list)

    def encoder_args(self) -> list[str]:
        args = ["-c:v", self.codec, "-crf", str(self.crf), "-g", str(self.gop),
                "-profile:v", self.profile, "-pix_fmt", self.pix_fmt,
                "-preset", self.preset]
        if self.bframes is not None:
            args += ["-bf", str(self.bframes)]
            if self.bframes == 0:
                # Without this x264 can still promote a B-frame to a reference,
                # which reintroduces the reorder delay `bframes=0` was chosen to
                # remove.
                args += ["-x264-params", "b-pyramid=0"]
        return args + list(self.extra_args)


@dataclass
class TranscodeResult:
    src: str
    dst: str
    status: str                     # ok | skipped_present | copied | failed
    src_res: tuple[int, int] | None = None
    dst_res: tuple[int, int] | None = None
    frames: int | None = None
    src_bytes: int | None = None
    dst_bytes: int | None = None
    seconds: float | None = None
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.status in ("ok", "skipped_present", "copied")


def _run(cmd: list[str], timeout: int) -> subprocess.CompletedProcess:
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired as exc:
        return subprocess.CompletedProcess(
            cmd, -9, exc.stdout or "",
            (exc.stderr or "") + f"\nffmpeg exceeded {timeout}s and was killed")


def probe(path: str | Path, ff: str | None = None) -> dict:
    """codec and resolution, read from the container header.

    ffmpeg exits with an error because no output was requested; the header dump
    it prints first is exactly what gets parsed. Cheaper than a full ffprobe and
    available in builds that ship no ffprobe at all.
    """
    ff = ff or ffmpeg_bin()
    err = _run([ff, "-hide_banner", "-nostdin", "-i", str(path)], 120).stderr
    m = _STREAM_RE.search(err)
    if not m:
        return dict(codec="", width=0, height=0)
    return dict(codec=m.group(1), width=int(m.group(2)), height=int(m.group(3)))


def count_frames(path: str | Path, ff: str | None = None, timeout: int = 1800) -> int:
    """Walk the stream and count. The container's `nb_frames` is metadata and is
    routinely wrong, and this number is the transcode's acceptance test."""
    ff = ff or ffmpeg_bin()
    r = _run([ff, "-v", "error", "-stats", "-nostdin", "-i", str(path),
              "-map", "0:v:0", "-f", "null", "-"], timeout)
    hits = _FRAME_RE.findall(r.stderr)
    return int(hits[-1]) if hits else 0


def target_dims(w: int, h: int, max_side: int | None) -> tuple[int, int]:
    """Cap the longer edge, keep the aspect ratio, keep both edges even.

    Even edges are required by 4:2:0 chroma subsampling. The long edge lands
    exactly on max_side and the short edge absorbs the rounding, so the two axes
    scale by very slightly different factors -- which matters if camera
    intrinsics are being rescaled alongside.
    """
    if not max_side or w <= 0 or h <= 0 or max(w, h) <= max_side:
        return w, h
    if w >= h:
        return max_side, max(2, round(h * max_side / w / 2) * 2)
    return max(2, round(w * max_side / h / 2) * 2), max_side


def transcode(src: str | Path, dst: str | Path, spec: TranscodeSpec | None = None,
              *, scratch: str | None = None, threads: int = 2,
              overwrite: bool = False, timeout: int = 1800,
              ff: str | None = None) -> TranscodeResult:
    """Encode `src` into `dst` under `spec`, verifying the frame count.

    Returns a result rather than raising, so a caller sweeping a corpus can
    tally failures and retry them instead of losing the whole run to one bad
    file.
    """
    spec = spec or TranscodeSpec()
    ff = ff or ffmpeg_bin()
    src, dst = Path(src), Path(dst)
    res = TranscodeResult(str(src), str(dst), "ok")

    if not src.is_file():
        return TranscodeResult(str(src), str(dst), "failed", error="source missing")
    if dst.exists() and dst.stat().st_size > 0 and not overwrite:
        return TranscodeResult(str(src), str(dst), "skipped_present",
                               dst_bytes=dst.stat().st_size)

    info = probe(src, ff)
    if not info["codec"]:
        return TranscodeResult(str(src), str(dst), "failed", error="unreadable source")
    w, h = info["width"], info["height"]
    dw, dh = target_dims(w, h, spec.max_side)
    res.src_res, res.dst_res = (w, h), (dw, dh)
    res.src_bytes = src.stat().st_size

    n_src = count_frames(src, ff, timeout)
    if n_src == 0:
        return TranscodeResult(str(src), str(dst), "failed", error="zero frames")
    res.frames = n_src

    dst.parent.mkdir(parents=True, exist_ok=True)
    scratch = scratch or os.environ.get("VLAFRAME_SCRATCH") or tempfile.gettempdir()
    Path(scratch).mkdir(parents=True, exist_ok=True)
    fd, tmp_s = tempfile.mkstemp(suffix=".mp4", dir=scratch)
    os.close(fd)
    tmp = Path(tmp_s)
    try:
        import time
        t0 = time.perf_counter()
        cmd = [ff, "-v", "error", "-nostdin", "-y"]
        if info["codec"] == "av1":
            # ffmpeg's default decoder for the av1 codec id is hwaccel-only and
            # fails outright on a CPU-only node.
            cmd += ["-c:v", "libdav1d"]
        cmd += ["-i", str(src), "-map", "0:v:0"]
        if (dw, dh) != (w, h):
            cmd += ["-vf", f"scale={dw}:{dh}:flags=bicubic"]
        cmd += spec.encoder_args()
        cmd += ["-vsync", "0", "-threads", str(threads), "-movflags", "+faststart",
                str(tmp)]
        r = _run(cmd, timeout)
        res.seconds = round(time.perf_counter() - t0, 3)
        if r.returncode != 0:
            return TranscodeResult(str(src), str(dst), "failed",
                                   error=r.stderr.strip()[-300:], seconds=res.seconds)

        n_dst = count_frames(tmp, ff, timeout)
        if n_dst != n_src:
            # Non-negotiable: downstream arrays index these frames positionally.
            return TranscodeResult(str(src), str(dst), "failed", frames=n_src,
                                   error=f"frame count changed {n_src} -> {n_dst}")
        res.dst_bytes = tmp.stat().st_size
        shutil.move(str(tmp), str(dst))
        tmp = None
    finally:
        if tmp is not None and tmp.exists():
            tmp.unlink()
    return res
