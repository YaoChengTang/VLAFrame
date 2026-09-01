"""On-disk index format (``.vfi``) — written once, then memory-mapped.

Layout, all little-endian, all offsets absolute from the start of the file::

    header      64 B          magic, version, counts, flags, section offsets
    paramtab    16 B * P      (offset, length) of each SPS/PPS blob
    paramblob   variable      the distinct avcC extradata, deduplicated
    root        variable      NUL-terminated path prefix; absent when off_root == 0
    strings     variable      NUL-terminated video paths, relative to root
    records     104 B * V     one fixed-size record per video: id -> record is O(1)
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

Nothing is cached on the Python heap
------------------------------------
Every accessor here reads the mmap on each call. That is deliberate: this file
is addressed by a DataLoader whose sampler is globally shuffled, so over one
epoch a worker touches essentially every video, and any per-video memo grows
without bound in *every* worker. On the 6.98 M video / 2.97 B frame corpus a
memo of unpacked records alone measured 581 B/video = 4.05 GB per worker. A
bounded cache would not help either -- the hit rate is structurally ~0, so an
LRU is "no cache, plus bookkeeping".

The one thing that made a memo tempting was ``path -> id``, which used to be a
dict built over the whole file on first lookup. ``HFLAG_PATHS_SORTED`` replaces
it: when the writer emitted the records in path order it says so in the header,
and :meth:`IndexFile.find` bisects the mmap instead (~20 probes, ~10 us, against
a ~20 ms decode). Indexes written before the flag existed are still read
correctly -- an unset flag falls back to the dict, which is what they always
did.

Paths are stored relative to a root
-----------------------------------
An index used to store each video's absolute path, and :meth:`IndexFile.find`
matches by exact bytes. That made the index a function of where the corpus was
mounted and of what the directories above the videos were called: renaming a
dataset directory, or mounting the same corpus under a different prefix on
another machine, made **every** lookup miss. Nothing failed at open time -- the
file is still a valid index -- so the first symptom was at read time, one frame
at a time, which a consumer substituting a black frame on a read error turns
into a silent corruption of its own inputs.

So the common prefix is stored once, in the ``root`` section, and the per-video
strings hold only the remainder. ``find`` strips the root from an absolute query
before bisecting, and accepts an already-relative one as is.

The root is *overridable at open time* (``IndexFile(path, root=...)``), and that
is the point: a corpus that moved needs a caller that passes the new prefix, not
a rebuilt index. What the index asserts is the shape below the root, which is
the part that actually describes the videos.

``off_root == 0`` means no root, i.e. the strings are absolute. That is exactly
how every index written before this existed reads, so they keep working
untouched: the field occupies 8 bytes that were reserved-and-zero in the
original header, and a zero there has always meant "absent".
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
# off_records, off_arrays, off_root
#
# `off_root` occupies the 8 bytes the original header reserved (`8x`), which
# struct.pack zero-filled. A zero there therefore means "no root, the strings
# are absolute" on every index ever written, and reading one needs no version
# check -- which is why this is not a version bump.
HEADER_FMT = "<8sIIIIQQQQQ"
HEADER_SIZE = struct.calcsize(HEADER_FMT)
# byte offset of the `flags` field, so it can be set in place on an existing
# file without rewriting it (see `seal_sorted`)
HEADER_FLAGS_OFF = struct.calcsize("<8sIII")

# path_off, path_len, w, h, param_set, n_frames, n_keys, flags,
# pos0, pts_base, pts_step, pad, off_size, off_disp, off_keys, off_ref,
# off_pos, off_pts
RECORD_FMT = "<QIHHIIIIQqiIQQQQQQ"
RECORD_SIZE = struct.calcsize(RECORD_FMT)
# just the (path_off, path_len) prefix -- the bisection reads nothing else
PATH_FMT = "<QI"

# header flags
HFLAG_PATHS_SORTED = 1 << 0     # records are ordered by UTF-8 path bytes

# per-video record flags
FLAG_EXPLICIT_POS = 1 << 0
FLAG_EXPLICIT_PTS = 1 << 1


@dataclass(frozen=True)
class VideoMeta:
    """One video's record, unpacked. Built fresh on every lookup, never memoed
    -- see the module docstring for why."""
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


def normalize_root(root: str | None) -> str:
    """A root with no trailing separator, so joining and stripping agree.

    ``""`` means "no root" everywhere in this package. A lone ``"/"`` would
    normalise to ``""`` under a plain rstrip and silently become that, so it is
    kept as-is; nothing else needs a special case.
    """
    if not root:
        return ""
    r = str(root)
    return r if r == os.sep else r.rstrip(os.sep)


def relative_to_root(path: str, root: str) -> str | None:
    """`path` as stored under `root`, or None when it does not live there.

    A plain prefix test rather than :func:`os.path.relpath`: relpath happily
    answers with ``../..`` for a path outside the root, and a stored ``../``
    would be a path that no longer identifies the video it came from.
    """
    if not root:
        return path
    if path == root:
        return ""
    sep = "" if root.endswith(os.sep) else os.sep
    prefix = root + sep
    return path[len(prefix):] if path.startswith(prefix) else None


# ───────────────────────────── writing ─────────────────────────────

class IndexWriter:
    """Collects videos, then emits the file in one pass.

    Array offsets have to be known before the fixed-size records are written, so
    the arrays are laid out first (on paper) and the records reference them.

    ``root`` is the prefix every video path shares and which is therefore worth
    storing once instead of V times. It is applied in :meth:`add`, so the sort
    in :meth:`write` orders the *stored* (relative) bytes -- which is what
    :meth:`IndexFile.find` bisects.
    """

    def __init__(self, root: str = "") -> None:
        self.root = normalize_root(root)
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
        stored = relative_to_root(str(path), self.root)
        if stored is None:
            raise ValueError(
                f"{path} is not under the index root {self.root!r}. Every video "
                f"in one index has to share its root, or the stored paths would "
                f"not mean the same thing.")
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
            path=stored, width=int(width), height=int(height),
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
        # Emit in path order and say so in the header: that is what lets a
        # reader bisect the mmap instead of building a path -> id dict over the
        # whole file. Sorting on the encoded bytes, not the str, because the
        # bisection compares the bytes as stored.
        self._videos.sort(key=lambda v: v["path"].encode())

        V, P = len(self._videos), len(self._params)
        blobs = b"".join(self._params)
        strings = b"".join(v["path"].encode() + b"\0" for v in self._videos)
        root_blob = (self.root.encode() + b"\0") if self.root else b""

        off_paramtab = HEADER_SIZE
        off_blobs = off_paramtab + 16 * P
        off_root = (off_blobs + len(blobs)) if root_blob else 0
        off_strings = off_blobs + len(blobs) + len(root_blob)
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
            fh.write(struct.pack(HEADER_FMT, MAGIC, VERSION, V, P,
                                 HFLAG_PATHS_SORTED,
                                 off_paramtab, off_strings, off_records,
                                 off_arrays, off_root))
            o = 0
            for b in self._params:
                fh.write(struct.pack("<QQ", off_blobs + o, len(b)))
                o += len(b)
            fh.write(blobs)
            fh.write(root_blob)
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
                    root=self.root,
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
    """Memory-mapped reader. Nothing is copied until a video is touched.

    ``root`` overrides the prefix stored in the file. That is what makes an
    index survive a corpus that moved: the stored root describes where the
    videos were when it was built, and a caller that knows better says so
    instead of rebuilding 25 GB of index to change one string.
    """

    def __init__(self, path: str, root: str | None = None):
        self.path = str(path)
        self._fh = open(self.path, "rb")
        self._mm = mmap.mmap(self._fh.fileno(), 0, access=mmap.ACCESS_READ)
        (magic, version, self.n_videos, self.n_params, self.flags,
         self._off_paramtab, self._off_strings, self._off_records,
         self._off_arrays, self._off_root) = struct.unpack_from(
            HEADER_FMT, self._mm, 0)
        if magic != MAGIC:
            raise ValueError(f"{path}: not a vlaframe index (magic {magic!r})")
        if version != VERSION:
            raise ValueError(f"{path}: index version {version}, this build reads {VERSION}")
        # Zero means absent, which is what every index written before the root
        # existed has in those bytes.
        self.stored_root = ""
        if self._off_root:
            end = self._mm.find(b"\0", self._off_root)
            if end < 0:
                raise ValueError(f"{path}: root string is not NUL-terminated")
            self.stored_root = self._mm[self._off_root:end].decode()
        self.root = normalize_root(self.stored_root if root is None else root)
        self._params: list[bytes] = []
        for i in range(self.n_params):
            o, ln = struct.unpack_from("<QQ", self._mm, self._off_paramtab + 16 * i)
            self._params.append(bytes(self._mm[o:o + ln]))
        self._by_path: dict[str, int] | None = None

    @property
    def paths_sorted(self) -> bool:
        """True when records are in path order, so :meth:`find` can bisect."""
        return bool(self.flags & HFLAG_PATHS_SORTED)

    @property
    def relative_paths(self) -> bool:
        """True when the stored strings are relative to a root."""
        return bool(self.stored_root)

    def meta(self, vid: int) -> VideoMeta:
        if not 0 <= vid < self.n_videos:
            raise IndexError(f"video id {vid} out of range (0..{self.n_videos-1})")
        (poff, plen, w, h, ps, nfr, nkeys, flags, pos0, pts_base, pts_step,
         _pad, o_size, o_disp, o_keys, o_ref, o_pos, o_pts) = struct.unpack_from(
            RECORD_FMT, self._mm, self._off_records + RECORD_SIZE * vid)
        path = self._mm[self._off_strings + poff:
                        self._off_strings + poff + plen].decode()
        return VideoMeta(vid, path, w, h, ps, nfr, nkeys, flags, pos0,
                         pts_base, pts_step,
                         o_size, o_disp, o_keys, o_ref, o_pos, o_pts)

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

    def path_bytes(self, vid: int) -> bytes:
        """One video's path, without unpacking the rest of its record.

        The bisection probes ~20 records per lookup and needs nothing but the
        path, so it reads the (offset, length) prefix rather than paying for a
        whole :class:`VideoMeta`.
        """
        poff, plen = struct.unpack_from(
            PATH_FMT, self._mm, self._off_records + RECORD_SIZE * vid)
        return self._mm[self._off_strings + poff:self._off_strings + poff + plen]

    def abspath(self, vid: int) -> str:
        """Where this video actually lives, root included.

        :attr:`VideoMeta.path` is the *stored* string, which is relative when
        the index has a root; opening a file needs this instead.
        """
        return self.join_root(self.meta(vid).path)

    def join_root(self, stored: str) -> str:
        """A stored path made absolute. Identity when the index has no root."""
        return os.path.join(self.root, stored) if self.root else stored

    def to_stored(self, path: str) -> str | None:
        """A caller's path in the form the strings section holds.

        Accepts either an absolute path under the root or an already-relative
        one. ``None`` means the path is not addressable by this index at all,
        which :meth:`find` turns into a miss.
        """
        if not self.root:
            return str(path)
        p = str(path)
        if not os.path.isabs(p):
            return p
        return relative_to_root(p, self.root)

    def find(self, path: str) -> int | None:
        """Video id for `path`, or None if this index does not cover it.

        `path` may be absolute (the root is stripped first) or already relative
        to the root. Bisects the mmap when the writer recorded that the records
        are in path order -- and note the ordering that matters is the *stored*
        one, so stripping before comparing is not an optimisation, it is what
        makes the comparison meaningful.

        Otherwise falls back to the path -> id dict, which is what every index
        written before ``HFLAG_PATHS_SORTED`` needs -- correctness never depends
        on the flag, only the cost does.

        Returning None rather than raising is what lets a caller probe several
        indexes for a path; :meth:`FrameReader.vid_of` turns it into a KeyError
        for the single-index case.
        """
        stored = self.to_stored(path)
        if stored is None:
            return None
        if not self.paths_sorted:
            return self.path_to_id().get(stored)
        target = stored.encode()
        lo, hi = 0, self.n_videos
        while lo < hi:
            mid = (lo + hi) // 2
            if self.path_bytes(mid) < target:
                lo = mid + 1
            else:
                hi = mid
        # lo is the first record >= target; it is a hit only if it is equal.
        if lo < self.n_videos and self.path_bytes(lo) == target:
            return lo
        return None

    def path_to_id(self) -> dict[str, int]:
        """Fallback lookup for indexes that are not in path order.

        One entry per video, so this is 4 GB on a 7 M video corpus and it lands
        in every worker that touches it. :meth:`find` avoids it whenever
        ``HFLAG_PATHS_SORTED`` is set; nothing in this package calls it
        otherwise.
        """
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


def seal_sorted(path: str) -> dict:
    """Set ``HFLAG_PATHS_SORTED`` on an existing index, in place, if it earns it.

    Indexes built before the flag existed are, in practice, already in path
    order -- ``iter_videos`` walks each directory with ``sorted(files)`` and
    ``build_index`` preserved that order. This checks rather than assumes
    (a wrong flag makes the bisection return *another video's* frames, which
    looks like a perfectly good image), and on success rewrites the 4-byte
    header field only. Nothing else in the file moves, so it costs one
    sequential pass over the records and a 4-byte write instead of a rebuild.

    Returns a dict with ``sealed`` (whether the flag is now set) and ``reason``.
    """
    with IndexFile(path) as f:
        if f.paths_sorted:
            return dict(path=str(path), videos=f.n_videos, sealed=True,
                        reason="already sealed")
        prev = None
        for vid in range(f.n_videos):
            cur = f.path_bytes(vid)
            if prev is not None and cur <= prev:
                return dict(path=str(path), videos=f.n_videos, sealed=False,
                            reason=f"not in path order at record {vid}: "
                                   f"{prev.decode(errors='replace')!r} then "
                                   f"{cur.decode(errors='replace')!r}")
            prev = cur
        videos = f.n_videos
        # Read-modify-write, not a bare store: the field holds every header flag,
        # and overwriting it with this one bit would clear any other that is set.
        flags = f.flags | HFLAG_PATHS_SORTED
    with open(path, "r+b") as fh:
        fh.seek(HEADER_FLAGS_OFF)
        fh.write(struct.pack("<I", flags))
    return dict(path=str(path), videos=videos, sealed=True, reason="sealed")
