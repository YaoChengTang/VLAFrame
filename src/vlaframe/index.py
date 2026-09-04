"""Function 2 — build the offline index.

The container is demuxed exactly once per video, here, and never again. What the
reader needs afterwards is: where each packet's bytes are, which packet carries a
given display frame, which packets are keyframes, and which packets nothing
refers to.

That last one is what makes the reader fast. H.264 marks it per NAL as
``nal_ref_idc``: a picture with ``nal_ref_idc == 0`` is referenced by nothing and
does not even advance ``frame_num``, so it can be dropped from a decode without
disturbing anything else. libavcodec will not do this for you -- after a seek it
decodes the entire prefix -- but with the flag recorded offline the reader can
feed only the references plus the target. Measured: 14 packets down to 8, and
164 KB of device reads down to 36 KB, at zero storage cost.

The flag is read out of the bitstream, never inferred from encoder settings.
x264's default b-pyramid promotes some B-frames to references, so a rule like
"B-frames are disposable" would corrupt exactly those.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Callable, Iterable, Iterator

import av
import numpy as np

from ._format import IndexWriter


def avcc_nal_len_size(extradata: bytes) -> int:
    """Length-prefix width for NALs, from the avcC record. Nearly always 4, but
    guessing wrong turns the NAL walk into noise and silently mislabels every
    reference frame, so it is read rather than assumed."""
    if len(extradata) >= 5 and extradata[0] == 1:
        return (extradata[4] & 0x03) + 1
    return 4


def is_reference_packet(buf: bytes | memoryview, nal_len_size: int) -> bool:
    """True when some other picture may refer to this one.

    Walks the packet's NALs and looks for a VCL slice (type 1 or 5) whose
    ``nal_ref_idc`` is non-zero.
    """
    i, n = 0, len(buf)
    while i + nal_len_size <= n:
        ln = int.from_bytes(buf[i:i + nal_len_size], "big")
        i += nal_len_size
        if ln <= 0 or i + ln > n:
            break
        b = buf[i]
        if (b & 0x1F) in (1, 5) and ((b >> 5) & 0x3) != 0:
            return True
        i += ln
    return False


def scan_video(path: str | Path) -> dict:
    """Demux once and return everything the index needs for this video."""
    with av.open(str(path)) as container:
        st = container.streams.video[0]
        extradata = bytes(st.codec_context.extradata or b"")
        width = int(st.codec_context.width)
        height = int(st.codec_context.height)
        nls = avcc_nal_len_size(extradata)
        pos, size, pts, key, ref = [], [], [], [], []
        for pk in container.demux(st):
            if pk.size == 0 or pk.pos is None or pk.pts is None:
                continue                      # demuxer flush packet, not payload
            pos.append(int(pk.pos))
            size.append(int(pk.size))
            pts.append(int(pk.pts))
            key.append(bool(pk.is_keyframe))
            ref.append(is_reference_packet(bytes(pk), nls))

    if not pos:
        raise ValueError(f"{path}: no decodable packets")
    pts_a = np.asarray(pts, dtype=np.int64)
    key_a = np.asarray(key, dtype=bool)
    if not key_a.any():
        raise ValueError(f"{path}: no keyframe")

    # Display rank -> packet index. Stable so equal timestamps keep decode order.
    by_display = np.argsort(pts_a, kind="stable").astype(np.uint32)
    keys = np.flatnonzero(key_a).astype(np.uint32)

    return dict(path=str(path), width=width, height=height, extradata=extradata,
                size=np.asarray(size, dtype=np.uint32),
                pos=np.asarray(pos, dtype=np.int64),
                pts=pts_a, by_display=by_display, keys=keys,
                ref=np.asarray(ref, dtype=bool))


def iter_videos(root: str | Path, suffixes: tuple[str, ...] = (".mp4",),
                video_dirs_only: bool = False) -> Iterator[str]:
    """Walk a tree for video files, skipping dotfiles.

    Dotfiles matter in practice: azcopy stages downloads as
    ``.azDownload-<uuid>-<name>`` and renames on completion, so a tree walked
    mid-transfer is full of paths that disappear underneath you.
    """
    for dirpath, _dirs, files in os.walk(str(root)):
        if video_dirs_only and not os.path.basename(dirpath).startswith("video"):
            continue
        for f in sorted(files):
            if f.startswith(".") or not f.endswith(suffixes):
                continue
            yield os.path.join(dirpath, f)


def build_index(videos: Iterable[str | Path], out_path: str | Path,
                *, root: str = "", on_error: str = "skip",
                progress: Callable[[int, int, str], None] | None = None) -> dict:
    """Index every video in `videos` into a single ``.vfi`` file.

    `root` is the prefix the stored paths are relative to. Every video has to
    live under it. Passing one is what keeps the index from being a function of
    where the corpus is mounted; leaving it empty stores absolute paths, which
    is what indexes built before roots existed did.

    `on_error='skip'` keeps going and reports the count, which is what a
    corpus-scale sweep wants; `'raise'` stops at the first bad file.
    """
    videos = list(videos)
    writer = IndexWriter(root)
    failed: list[tuple[str, str]] = []
    for i, path in enumerate(videos, 1):
        try:
            v = scan_video(path)
        except Exception as exc:
            if on_error == "raise":
                raise
            failed.append((str(path), f"{type(exc).__name__}: {exc}"))
            continue
        writer.add(v["path"], v["width"], v["height"],
                   writer.param_set(v["extradata"]),
                   v["size"], v["pos"], v["pts"], v["by_display"],
                   v["keys"], v["ref"])
        if progress:
            progress(i, len(videos), str(path))
    stats = writer.write(str(out_path))
    stats["failed"] = len(failed)
    stats["failures"] = failed[:20]
    return stats


def build_index_from_root(root: str | Path, out_path: str | Path,
                          *, relative: bool = True, **kw) -> dict:
    """Convenience wrapper: walk `root` for mp4 and index all of them.

    ``relative`` stores the paths under `root` rather than absolute, which is
    almost always what is wanted -- the tree being walked *is* the shared
    prefix. Pass ``relative=False`` to reproduce the pre-root behaviour.
    """
    limit = kw.pop("limit", 0)
    files = list(iter_videos(root, video_dirs_only=kw.pop("video_dirs_only", False)))
    if limit:
        files = files[:limit]
    return build_index(files, out_path, root=str(root) if relative else "", **kw)
