"""On-disk index format (``.vfi``) — written once, then memory-mapped.

Layout, all little-endian, all offsets absolute from the start of the file::

    header      64 B          magic, version, counts, section offsets
    paramtab    16 B * P      (offset, length) of each SPS/PPS blob
    paramblob   variable      the distinct avcC extradata, deduplicated
    strings     variable      NUL-terminated video paths
    records     80 B * V      one fixed-size record per video: id -> record is O(1)
    arrays      variable      the per-frame arrays, 8-byte aligned

What is deliberately NOT stored, and why
----------------------------------------
``pts``   Presentation timestamps are arithmetic in DISPLAY order on every video
          measured (200/200), so only a base and a step are stored; the value for
          a decode-order packet is ``base + display_rank(packet) * step``. Any
          video that is not arithmetic stores an explicit array and sets
          ``FLAG_EXPLICIT_PTS``. Saves 8 B/frame.

          Real timestamps are NOT optional. Feeding synthetic decode-order
          integers works with the software decoder -- it reorders to display
          order and carries each value with its own picture -- but ``h264_cuvid``
          assigns timestamps to its reordered output POSITIONALLY, so synthetic
          values silently pair every frame with the wrong picture. Measured
          63/111 exact, with differences up to 244 per channel.

``pos``   Packets sit back-to-back inside ``mdat``, so a packet's byte offset is
          the first packet's offset plus the sizes preceding it, accumulated
          while walking the (short) keyframe-to-target range. Checked per video
          at build time; any video that violates it stores an explicit array and
          sets ``FLAG_EXPLICIT_POS``. Measured 300/300 contiguous on the sample
          corpus. Saves another 8 B/frame.

``prev_key``  Derived by bisecting the keyframe list, which is ~1/GOP the length
          of a per-frame array. Saves 4 B/frame.

Net: ~8.3 B/frame instead of 29, so a 2.6-billion-frame corpus indexes to about
22 GB rather than 76 GB. Being mmap-ed, a worker faults in only the pages of the
videos it actually samples.
"""
from __future__ import annotations

import mmap
import os
import struct
from dataclasses import dataclass

import numpy as np

MAGIC = b"VLAFRME1"
VERSION = 1

# magic, version, n_videos, n_params, flags, off_paramtab, off_strings,
# off_records, off_arrays, reserved
HEADER_FMT = "<8sIIIIQQQQ8x"
HEADER_SIZE = struct.calcsize(HEADER_FMT)

# path_off, path_len, w, h, param_set, n_frames, n_keys, flags,
# pos0, pts_base, pts_step, pad, off_size, off_disp, off_keys, off_ref,
# off_pos, off_pts
RECORD_FMT = "<QIHHIIIIQqiIQQQQQQ"
RECORD_SIZE = struct.calcsize(RECORD_FMT)

FLAG_EXPLICIT_POS = 1 << 0
FLAG_EXPLICIT_PTS = 1 << 1


@dataclass(frozen=True)
class VideoMeta:
    """One video's record, unpacked. Cheap to build, cached by the reader."""
    vid: int
    path: str
    width: int
    height: int
    param_set: int
    n_frames: int
    n_keys: int
    flags: int
    pos0: int
    pts_base: int
    pts_step: int
    off_size: int
    off_disp: int
    off_keys: int
    off_ref: int
    off_pos: int
    off_pts: int


def _align(x: int, a: int = 8) -> int:
    return (x + a - 1) // a * a


# ───────────────────────────── writing ─────────────────────────────

class IndexWriter:
    """Collects videos, then emits the file in one pass.

    Array offsets have to be known before the fixed-size records are written, so
    the arrays are laid out first (on paper) and the records reference them.
    """

    def __init__(self) -> None:
        self._params: list[bytes] = []
        self._param_ids: dict[bytes, int] = {}
        self._videos: list[dict] = []

    def param_set(self, extradata: bytes) -> int:
        """Intern one SPS/PPS blob. Real corpora have a handful, not thousands."""
        pid = self._param_ids.get(extradata)
        if pid is None:
            pid = len(self._params)
            self._param_ids[extradata] = pid
            self._params.append(extradata)
        return pid

    def add(self, path: str, width: int, height: int, param_set: int,
            size: np.ndarray, pos: np.ndarray, pts: np.ndarray,
            by_display: np.ndarray, keys: np.ndarray, ref: np.ndarray) -> None:
        size = np.ascontiguousarray(size, dtype=np.uint32)
        pos = np.ascontiguousarray(pos, dtype=np.int64)
        pts = np.ascontiguousarray(pts, dtype=np.int64)
        expected = int(pos[0]) + np.concatenate(
            [[0], np.cumsum(size[:-1].astype(np.int64))])
        contiguous = bool(np.array_equal(pos, expected))

        # pts in display order: arithmetic on every video measured, so keep a
        # base and a step and reconstruct. Validated here rather than assumed --
        # a wrong timestamp pairs a frame with the wrong picture on the GPU
        # decoder, silently.
        disp_pts = pts[by_display.astype(np.int64)]
        step = int(disp_pts[1] - disp_pts[0]) if disp_pts.size > 1 else 0
        base = int(disp_pts[0])
        uniform = disp_pts.size < 2 or bool(np.array_equal(
            disp_pts, base + step * np.arange(disp_pts.size, dtype=np.int64)))
        self._videos.append(dict(
            path=path, width=int(width), height=int(height),
            param_set=int(param_set), n_frames=int(size.size),
            size=size,
            disp=np.ascontiguousarray(by_display, dtype=np.uint32),
            keys=np.ascontiguousarray(keys, dtype=np.uint32),
            ref=np.packbits(np.asarray(ref, dtype=bool), bitorder="little"),
            pos0=int(pos[0]),
            pos=None if contiguous else pos,
            pts_base=base, pts_step=step,
            pts=None if uniform else pts,
        ))

    def write(self, out_path: str) -> dict:
        V, P = len(self._videos), len(self._params)
        blobs = b"".join(self._params)
        strings = b"".join(v["path"].encode() + b"\0" for v in self._videos)

        off_paramtab = HEADER_SIZE
        off_blobs = off_paramtab + 16 * P
        off_strings = off_blobs + len(blobs)
        off_records = _align(off_strings + len(strings))
        off_arrays = _align(off_records + RECORD_SIZE * V)

        # lay out the arrays and remember where each landed
        cursor = off_arrays
        for v in self._videos:
            for name in ("size", "disp", "keys", "ref", "pos", "pts"):
                a = v[name]
                if a is None:
                    v[name + "_off"] = 0
                    continue
                v[name + "_off"] = cursor
                cursor = _align(cursor + a.nbytes)

        soff, c = [], 0
        for v in self._videos:
            soff.append(c)
            c += len(v["path"].encode()) + 1

        with open(out_path, "wb") as fh:
            fh.write(struct.pack(HEADER_FMT, MAGIC, VERSION, V, P, 0,
                                 off_paramtab, off_strings, off_records, off_arrays))
            o = 0
            for b in self._params:
                fh.write(struct.pack("<QQ", off_blobs + o, len(b)))
                o += len(b)
            fh.write(blobs)
            fh.write(strings)
            _pad_to(fh, off_records)
            for i, v in enumerate(self._videos):
                fh.write(struct.pack(
                    RECORD_FMT, soff[i], len(v["path"].encode()),
                    v["width"], v["height"], v["param_set"], v["n_frames"],
                    int(v["keys"].size),
                    (0 if v["pos"] is None else FLAG_EXPLICIT_POS)
                    | (0 if v["pts"] is None else FLAG_EXPLICIT_PTS),
                    v["pos0"], v["pts_base"], v["pts_step"], 0,
                    v["size_off"], v["disp_off"], v["keys_off"],
                    v["ref_off"], v["pos_off"], v["pts_off"]))
            _pad_to(fh, off_arrays)
            for v in self._videos:
                for name in ("size", "disp", "keys", "ref", "pos", "pts"):
                    a = v[name]
                    if a is None:
                        continue
                    _pad_to(fh, v[name + "_off"])
                    fh.write(a.tobytes())

        total = os.path.getsize(out_path)
        frames = sum(v["n_frames"] for v in self._videos)
        return dict(videos=V, frames=frames, param_sets=P, bytes=total,
                    bytes_per_frame=total / max(frames, 1),
                    explicit_pos=sum(1 for v in self._videos if v["pos"] is not None),
                    explicit_pts=sum(1 for v in self._videos if v["pts"] is not None))


def _pad_to(fh, target: int) -> None:
    cur = fh.tell()
    if cur > target:
        raise RuntimeError(f"index layout overflow: at {cur}, expected <= {target}")
    if cur < target:
        fh.write(b"\0" * (target - cur))


# ───────────────────────────── reading ─────────────────────────────

class IndexFile:
    """Memory-mapped reader. Nothing is copied until a video is touched."""

    def __init__(self, path: str):
        self.path = str(path)
        self._fh = open(self.path, "rb")
        self._mm = mmap.mmap(self._fh.fileno(), 0, access=mmap.ACCESS_READ)
        (magic, version, self.n_videos, self.n_params, _flags,
         self._off_paramtab, self._off_strings, self._off_records,
         self._off_arrays) = struct.unpack_from(HEADER_FMT, self._mm, 0)
        if magic != MAGIC:
            raise ValueError(f"{path}: not a vlaframe index (magic {magic!r})")
        if version != VERSION:
            raise ValueError(f"{path}: index version {version}, this build reads {VERSION}")
        self._params: list[bytes] = []
        for i in range(self.n_params):
            o, ln = struct.unpack_from("<QQ", self._mm, self._off_paramtab + 16 * i)
            self._params.append(bytes(self._mm[o:o + ln]))
        self._meta: dict[int, VideoMeta] = {}
        self._by_path: dict[str, int] | None = None

    def meta(self, vid: int) -> VideoMeta:
        m = self._meta.get(vid)
        if m is not None:
            return m
        if not 0 <= vid < self.n_videos:
            raise IndexError(f"video id {vid} out of range (0..{self.n_videos-1})")
        (poff, plen, w, h, ps, nfr, nkeys, flags, pos0, pts_base, pts_step,
         _pad, o_size, o_disp, o_keys, o_ref, o_pos, o_pts) = struct.unpack_from(
            RECORD_FMT, self._mm, self._off_records + RECORD_SIZE * vid)
        path = self._mm[self._off_strings + poff:
                        self._off_strings + poff + plen].decode()
        m = VideoMeta(vid, path, w, h, ps, nfr, nkeys, flags, pos0,
                      pts_base, pts_step,
                      o_size, o_disp, o_keys, o_ref, o_pos, o_pts)
        self._meta[vid] = m
        return m

    def extradata(self, param_set: int) -> bytes:
        return self._params[param_set]

    def sizes(self, m: VideoMeta) -> np.ndarray:
        return np.frombuffer(self._mm, np.uint32, m.n_frames, m.off_size)

    def by_display(self, m: VideoMeta) -> np.ndarray:
        return np.frombuffer(self._mm, np.uint32, m.n_frames, m.off_disp)

    def keys(self, m: VideoMeta) -> np.ndarray:
        return np.frombuffer(self._mm, np.uint32, m.n_keys, m.off_keys)

    def refbits(self, m: VideoMeta) -> np.ndarray:
        return np.frombuffer(self._mm, np.uint8, (m.n_frames + 7) // 8, m.off_ref)

    def positions(self, m: VideoMeta) -> np.ndarray | None:
        """Explicit byte offsets, present only for the videos whose packets are
        not laid out back-to-back. `None` means derive from pos0 + sizes."""
        if not (m.flags & FLAG_EXPLICIT_POS):
            return None
        return np.frombuffer(self._mm, np.int64, m.n_frames, m.off_pos)

    def timestamps(self, m: VideoMeta) -> np.ndarray | None:
        """Explicit presentation timestamps, present only when they are not
        arithmetic in display order. `None` means derive from base + step."""
        if not (m.flags & FLAG_EXPLICIT_PTS):
            return None
        return np.frombuffer(self._mm, np.int64, m.n_frames, m.off_pts)

    def path_to_id(self) -> dict[str, int]:
        """Built lazily: a corpus-scale index has millions of entries and most
        callers address videos by id."""
        if self._by_path is None:
            self._by_path = {self.meta(i).path: i for i in range(self.n_videos)}
        return self._by_path

    def close(self) -> None:
        try:
            self._mm.close()
        finally:
            self._fh.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
