"""Function 4 — combine index shards into one ``.vfi``, byte for byte.

Building a corpus-scale index is a parallel job: ``IndexWriter`` buffers
everything in memory before it writes, so a single pass over seven million
videos does not fit, and the practical answer is to index in shards. That is a
property of *building*, and it should not survive into reading -- a consumer
that has to know which shard holds a path has to know the sharding convention,
the manifest schema, and how to route, and every later change to any of those
becomes a change to the consumer.

So the shards are an intermediate product. This merges them back:

    merge_indexes(glob.glob("shards/Leaf__*.vfi"), "Leaf.vfi", verify=True)

What makes it safe to do at all is that it moves bytes rather than rebuilding.
The per-frame arrays are copied from the source mmap to the output unchanged,
and every field of a video's record is carried across verbatim except the two
that *must* change: the parameter-set id, which is renumbered against the merged
table, and the six array offsets, which point at the new layout.

**Record flags are carried, never recomputed.** ``FLAG_EXPLICIT_POS`` and
``FLAG_EXPLICIT_PTS`` are conclusions the indexer reached by *measuring that
video* -- whether its packets are back-to-back, whether its timestamps are
arithmetic in display order. Merging has no packets to measure, so recomputing
would mean guessing at an assumption instead of inheriting a verified one, and
the failure that produces is a frame paired with the wrong timestamp: nothing
about the image looks wrong. The two traps this package already documents
(synthetic timestamps, NV12 vs YUV420P) are both of that kind.

The output is written in path order with ``HFLAG_PATHS_SORTED`` set, so a merged
index is also a bisectable one.
"""
from __future__ import annotations

import os
import struct
import tempfile
from typing import Callable, Iterable

from ._format import (FLAG_EXPLICIT_POS, FLAG_EXPLICIT_PTS, HEADER_FMT,
                      HEADER_SIZE, HFLAG_PATHS_SORTED, MAGIC, RECORD_FMT,
                      RECORD_SIZE, VERSION, IndexFile, _align, _pad_to)

# One planned video. A tuple rather than a dataclass or dict because there is
# one per video in the corpus and the planning pass holds them all: 6.98 M of
# these is ~200 B each, and the dict form measured three times that.
#   0 path      1 src        2 width     3 height    4 param_set (new)
#   5 n_frames  6 n_keys     7 flags     8 pos0      9 pts_base   10 pts_step
#   11..16 source offsets of size, disp, keys, ref, pos, pts
_ARRAYS = ("size", "disp", "keys", "ref", "pos", "pts")


def _array_sizes(n_frames: int, n_keys: int, flags: int) -> tuple[int, ...]:
    """Bytes of each of the six arrays, derived from the record alone.

    Nothing is read to work this out, which is what keeps planning cheap: the
    arrays are the whole file, and this pass never touches them.
    """
    return (
        4 * n_frames,                                   # size   uint32
        4 * n_frames,                                   # disp   uint32
        4 * n_keys,                                     # keys   uint32
        (n_frames + 7) // 8,                            # ref    packed bits
        8 * n_frames if flags & FLAG_EXPLICIT_POS else 0,
        8 * n_frames if flags & FLAG_EXPLICIT_PTS else 0,
    )


def merge_indexes(inputs: Iterable[str], out_path: str, *,
                  on_duplicate: str = "error", verify: bool = False,
                  progress: Callable[[int, int, str], None] | None = None) -> dict:
    """Merge `inputs` into one ``.vfi`` at `out_path`.

    `on_duplicate` decides what happens when two shards claim the same video
    path: ``error`` (the default) refuses and names both shards, ``first`` and
    ``last`` keep one silently. The default is to refuse because a duplicate
    means the shards were not a partition, and quietly picking one would hide
    whichever mistake produced that.

    `verify` re-opens the output and every input afterwards and compares them
    field by field and byte by byte -- see :func:`verify_merge`. Seconds, no
    decoding, and it is the check that makes "byte for byte" a claim rather
    than an intention.
    """
    if on_duplicate not in ("error", "first", "last"):
        raise ValueError(f"on_duplicate must be error/first/last, got {on_duplicate!r}")

    inputs = sorted(str(p) for p in inputs)             # deterministic order
    if not inputs:
        raise ValueError("no input indexes")
    real_out = os.path.realpath(out_path)
    for p in inputs:
        if os.path.realpath(p) == real_out:
            raise ValueError(f"{out_path} is also an input; that would read and "
                             f"write the same file")

    # ── pass 1: plan ──────────────────────────────────────────────────
    srcs: list[IndexFile] = []
    blobs: list[bytes] = []
    blob_ids: dict[bytes, int] = {}
    plan: list[tuple] = []
    seen: dict[bytes, int] = {}                          # path -> index in plan
    duplicates = 0
    empty = 0
    n_in_params = 0

    try:
        for si, path in enumerate(inputs):
            f = IndexFile(path)                          # validates magic/version
            srcs.append(f)
            n_in_params += f.n_params
            if f.n_videos == 0:
                empty += 1
                continue
            remap = {}
            for old in range(f.n_params):
                blob = f.extradata(old)
                new = blob_ids.get(blob)
                if new is None:
                    new = len(blobs)
                    blob_ids[blob] = new
                    blobs.append(blob)
                remap[old] = new
            for vid in range(f.n_videos):
                m = f.meta(vid)
                if m.n_keys == 0:
                    raise ValueError(
                        f"{path}: video {vid} ({m.path}) has no keyframe; the "
                        f"index is malformed and merging would carry that "
                        f"forward silently")
                key = m.path.encode()
                row = (key, si, m.width, m.height, remap[m.param_set],
                       m.n_frames, m.n_keys, m.flags, m.pos0,
                       m.pts_base, m.pts_step,
                       m.off_size, m.off_disp, m.off_keys,
                       m.off_ref, m.off_pos, m.off_pts)
                prev = seen.get(key)
                if prev is None:
                    seen[key] = len(plan)
                    plan.append(row)
                    continue
                duplicates += 1
                if on_duplicate == "error":
                    raise ValueError(
                        f"{m.path} is in two inputs:\n  {inputs[plan[prev][1]]}"
                        f"\n  {path}\nPass on_duplicate='first' or 'last' if "
                        f"that is intended.")
                if on_duplicate == "last":
                    plan[prev] = row
            if progress:
                progress(si + 1, len(inputs), path)

        if not plan:
            raise ValueError(f"all {len(inputs)} input(s) are empty")

        # Path order is what lets the reader bisect instead of building a dict.
        plan.sort(key=lambda r: r[0])

        # ── layout, in the order IndexWriter.write() emits ─────────────
        V, P = len(plan), len(blobs)
        blob_bytes = b"".join(blobs)
        strings = b"".join(r[0] + b"\0" for r in plan)
        off_paramtab = HEADER_SIZE
        off_blobs = off_paramtab + 16 * P
        off_strings = off_blobs + len(blob_bytes)
        off_records = _align(off_strings + len(strings))
        off_arrays = _align(off_records + RECORD_SIZE * V)

        dst_offs: list[tuple[int, ...]] = []
        cursor = off_arrays
        for r in plan:
            sizes = _array_sizes(r[5], r[6], r[7])
            here = []
            for nbytes in sizes:
                if nbytes == 0:
                    here.append(0)
                    continue
                here.append(cursor)
                cursor = _align(cursor + nbytes)
            dst_offs.append(tuple(here))

        soff, c = [], 0
        for r in plan:
            soff.append(c)
            c += len(r[0]) + 1

        # ── pass 2: write ─────────────────────────────────────────────
        # Into a temporary file in the destination directory, then renamed, so
        # a failure halfway leaves no file with a valid magic for something
        # else to pick up. Same shape as transcode.py.
        out_dir = os.path.dirname(os.path.abspath(out_path)) or "."
        os.makedirs(out_dir, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=out_dir, prefix=".merge-", suffix=".vfi")
        os.close(fd)
        # mkstemp creates 0600 and os.replace preserves it, so without this the
        # output is readable only by whoever ran the merge. An index built by an
        # administrator over a shared corpus and then unreadable by the training
        # job is a plausible and annoying way to lose an afternoon; match what
        # open(path, "wb") would have produced instead.
        umask = os.umask(0)
        os.umask(umask)
        os.chmod(tmp, 0o666 & ~umask)
        try:
            with open(tmp, "wb") as fh:
                fh.write(struct.pack(HEADER_FMT, MAGIC, VERSION, V, P,
                                     HFLAG_PATHS_SORTED, off_paramtab,
                                     off_strings, off_records, off_arrays))
                o = 0
                for b in blobs:
                    fh.write(struct.pack("<QQ", off_blobs + o, len(b)))
                    o += len(b)
                fh.write(blob_bytes)
                fh.write(strings)
                _pad_to(fh, off_records)
                for i, r in enumerate(plan):
                    d = dst_offs[i]
                    fh.write(struct.pack(
                        RECORD_FMT, soff[i], len(r[0]), r[2], r[3], r[4],
                        r[5], r[6], r[7], r[8], r[9], r[10], 0,
                        d[0], d[1], d[2], d[3], d[4], d[5]))
                _pad_to(fh, off_arrays)
                for i, r in enumerate(plan):
                    mm = srcs[r[1]]._mm
                    sizes = _array_sizes(r[5], r[6], r[7])
                    for j in range(6):
                        if sizes[j] == 0:
                            continue
                        _pad_to(fh, dst_offs[i][j])
                        src_off = r[11 + j]
                        # An explicit slice copy, not a numpy view: a view would
                        # keep the source mmap exported and its close() would
                        # raise BufferError.
                        fh.write(mm[src_off:src_off + sizes[j]])
            os.replace(tmp, out_path)
        except BaseException:
            if os.path.exists(tmp):
                os.unlink(tmp)
            raise

        total = os.path.getsize(out_path)
        frames = sum(r[5] for r in plan)
        stats = dict(
            videos=V, frames=frames, param_sets=P, bytes=total,
            bytes_per_frame=total / max(frames, 1),
            inputs=len(inputs), empty_inputs=empty, duplicates=duplicates,
            input_param_sets=n_in_params,
            explicit_pos=sum(1 for r in plan if r[7] & FLAG_EXPLICIT_POS),
            explicit_pts=sum(1 for r in plan if r[7] & FLAG_EXPLICIT_PTS))
    finally:
        for f in srcs:
            try:
                f.close()
            except Exception:
                pass

    if verify:
        stats["verified"] = verify_merge(inputs, out_path)
    return stats


def verify_merge(inputs: Iterable[str], out_path: str) -> dict:
    """Re-open the output and every input and compare them exhaustively.

    Every video, not a sample, and every byte of every array, not a length or a
    checksum of the record. A merge that lands one array a few bytes off still
    produces a file that opens, reports the right counts, and decodes into
    plausible images for most videos -- the same failure mode the correctness
    gate exists for, one layer down. Sampling would pass it.

    Raises on the first disagreement; returns counts on success.
    """
    inputs = sorted(str(p) for p in inputs)
    out = IndexFile(out_path)
    checked = arrays = 0
    # Which output records an input accounted for. A bitset rather than a set of
    # paths because this runs on merges of millions of videos, and it has to be
    # per-output-record rather than per-input-video: with on_duplicate=first or
    # last, several input videos legitimately map to one output record.
    covered = bytearray((out.n_videos + 7) // 8)
    try:
        if not out.paths_sorted:
            raise ValueError(f"{out_path}: merged output is not marked sorted")
        for path in inputs:
            src = IndexFile(path)
            try:
                for vid in range(src.n_videos):
                    a = src.meta(vid)
                    got = out.find(a.path)
                    if got is None:
                        raise ValueError(f"{a.path} (from {path}) is missing "
                                         f"from {out_path}")
                    b = out.meta(got)
                    if (covered[got >> 3] >> (got & 7)) & 1:
                        # Another input already accounted for this record. Under
                        # on_duplicate=first/last only one of the two survives,
                        # so the loser is expected to differ -- comparing it
                        # would fail a merge that did exactly what was asked.
                        continue
                    for field in ("path", "width", "height", "n_frames",
                                  "n_keys", "flags", "pos0", "pts_base",
                                  "pts_step"):
                        if getattr(a, field) != getattr(b, field):
                            raise ValueError(
                                f"{a.path}: {field} is {getattr(b, field)!r} in "
                                f"the merge, {getattr(a, field)!r} in {path}")
                    if src.extradata(a.param_set) != out.extradata(b.param_set):
                        raise ValueError(f"{a.path}: parameter set content "
                                         f"differs after renumbering")
                    sizes = _array_sizes(a.n_frames, a.n_keys, a.flags)
                    a_offs = (a.off_size, a.off_disp, a.off_keys,
                              a.off_ref, a.off_pos, a.off_pts)
                    b_offs = (b.off_size, b.off_disp, b.off_keys,
                              b.off_ref, b.off_pos, b.off_pts)
                    for j, nbytes in enumerate(sizes):
                        if nbytes == 0:
                            if b_offs[j] != 0:
                                raise ValueError(
                                    f"{a.path}: {_ARRAYS[j]} is absent in "
                                    f"{path} but present in the merge")
                            continue
                        if b_offs[j] % 8:
                            raise ValueError(
                                f"{a.path}: {_ARRAYS[j]} lands at {b_offs[j]}, "
                                f"not 8-byte aligned")
                        if (src._mm[a_offs[j]:a_offs[j] + nbytes]
                                != out._mm[b_offs[j]:b_offs[j] + nbytes]):
                            raise ValueError(
                                f"{a.path}: {_ARRAYS[j]} differs byte for byte")
                        arrays += 1
                    covered[got >> 3] |= 1 << (got & 7)
                    checked += 1
            finally:
                src.close()
        missing = out.n_videos - checked
        if missing:
            raise ValueError(f"{out_path} holds {out.n_videos} videos but the "
                             f"inputs account for only {checked}; {missing} "
                             f"record(s) came from nowhere")
    finally:
        out.close()
    return dict(videos=checked, arrays=arrays, inputs=len(inputs))
