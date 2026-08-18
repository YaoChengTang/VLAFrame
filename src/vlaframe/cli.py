"""Command line: ``vlaframe transcode | index | info | seal | get | bench``.

Thin wrappers over the library. Everything the CLI does is available as a
function; the CLI exists so a corpus sweep does not need a driver script.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import random
import sys
import time
from pathlib import Path

from . import __version__
from ._format import seal_sorted
from .index import build_index, iter_videos
from .reader import FrameReader
from .transcode import TranscodeSpec, probe, transcode


def _cmd_transcode(a) -> int:
    spec = TranscodeSpec(gop=a.gop, crf=a.crf, bframes=a.bframes,
                         profile=a.profile, pix_fmt=a.pix_fmt, preset=a.preset,
                         max_side=a.max_side)
    src_root = Path(a.src)
    if src_root.is_file():
        pairs = [(str(src_root), a.dst)]
    else:
        pairs = []
        for p in iter_videos(src_root, video_dirs_only=a.video_dirs_only):
            rel = os.path.relpath(p, src_root)
            pairs.append((p, os.path.join(a.dst, rel)))
        if a.limit:
            pairs = pairs[:a.limit]
    print(f"transcode {len(pairs)} file(s)  gop={spec.gop} crf={spec.crf} "
          f"bframes={spec.bframes} profile={spec.profile} "
          f"max_side={spec.max_side}", flush=True)

    ok = failed = skipped = 0
    src_mb = dst_mb = 0.0
    t0 = time.perf_counter()
    for i, (s, d) in enumerate(pairs, 1):
        r = transcode(s, d, spec, scratch=a.scratch, threads=a.threads,
                      overwrite=a.overwrite)
        if r.status == "failed":
            failed += 1
            if failed <= 10:
                print(f"  FAIL {r.error}  {s}", flush=True)
        elif r.status == "skipped_present":
            skipped += 1
        else:
            ok += 1
            src_mb += (r.src_bytes or 0) / 1e6
            dst_mb += (r.dst_bytes or 0) / 1e6
        if i % 100 == 0:
            print(f"  [{i}/{len(pairs)}] ok={ok} skipped={skipped} failed={failed}",
                  flush=True)
    wall = time.perf_counter() - t0
    print(f"\nok={ok} skipped={skipped} failed={failed} in {wall:.0f}s")
    if src_mb:
        print(f"size {src_mb:.1f} MB -> {dst_mb:.1f} MB  ({dst_mb/src_mb:.3f}x)")
    return 1 if failed else 0


def _cmd_index(a) -> int:
    root = Path(a.root)
    files = [str(root)] if root.is_file() else list(
        iter_videos(root, video_dirs_only=a.video_dirs_only))
    if a.limit:
        files = files[:a.limit]
    print(f"indexing {len(files):,} file(s) under {root}", flush=True)
    t0 = time.perf_counter()

    def progress(i, n, path):
        if i % 500 == 0:
            print(f"  [{i}/{n}] {time.perf_counter()-t0:.0f}s", flush=True)

    st = build_index(files, a.out, progress=progress)
    print(f"\nindexed {st['videos']:,} videos, {st['frames']:,} frames, "
          f"{st['param_sets']} parameter set(s), {st['failed']} failed")
    print(f"index {st['bytes']/1e6:.1f} MB = {st['bytes_per_frame']:.2f} B/frame; "
          f"{st['explicit_pos']} video(s) needed explicit byte offsets; "
          f"built in {time.perf_counter()-t0:.0f}s")
    for p, e in st["failures"]:
        print(f"  FAILED {e}  {p}")
    return 0


def _cmd_info(a) -> int:
    with FrameReader(a.index) as r:
        n = len(r)
        frames = sum(r.index.meta(i).n_frames for i in range(n))
        print(f"{a.index}")
        print(f"  videos       : {n:,}")
        print(f"  frames       : {frames:,}")
        print(f"  parameter sets: {r.index.n_params}")
        lookup = "bisect (paths sorted)" if r.index.paths_sorted else \
            "dict (paths not sorted -- see `vlaframe seal`)"
        print(f"  path lookup  : {lookup}")
        print(f"  file size    : {os.path.getsize(a.index)/1e6:.1f} MB "
              f"({os.path.getsize(a.index)/max(frames,1):.2f} B/frame)")
        for i in range(min(a.show, n)):
            m = r.index.meta(i)
            print(f"    [{i}] {m.width}x{m.height} {m.n_frames:>7} frames  "
                  f"{m.n_keys:>5} keys  {m.path}")
    return 0


def _cmd_seal(a) -> int:
    """Upgrade indexes written before `HFLAG_PATHS_SORTED` existed.

    Every one of them checked so far was already in path order, so this is a
    verified 4-byte header write rather than a rebuild. Unsealed indexes still
    read correctly -- they just fall back to the path -> id dict, which is the
    memory this flag exists to avoid.
    """
    paths = []
    for p in a.index:
        paths.extend(sorted(glob.glob(p)) if any(c in p for c in "*?[") else [p])
    if not paths:
        print("no index files matched", file=sys.stderr)
        return 1
    sealed = failed = 0
    for p in paths:
        r = seal_sorted(p)
        if r["sealed"]:
            sealed += 1
            if a.verbose:
                print(f"  {r['reason']:14s} {r['videos']:>8,} videos  {p}", flush=True)
        else:
            failed += 1
            print(f"  NOT SEALED {p}\n    {r['reason']}", flush=True)
    print(f"\nsealed {sealed}/{len(paths)}"
          f"{f', {failed} not in path order' if failed else ''}")
    return 1 if failed else 0


def _cmd_get(a) -> int:
    with FrameReader(a.index, backend=a.backend, threads=a.threads) as r:
        video = int(a.video) if a.video.isdigit() else a.video
        img = r.rgb(video, a.frame)
        print(f"frame {a.frame} of {video}: {img.shape} {img.dtype}")
        if a.out:
            try:
                from PIL import Image
                Image.fromarray(img).save(a.out)
                print(f"wrote {a.out}")
            except ImportError:
                print("install pillow to write an image file", file=sys.stderr)
                return 1
    return 0


def _cmd_bench(a) -> int:
    rnd = random.Random(a.seed)
    with FrameReader(a.index, backend=a.backend, threads=a.threads) as r:
        n = len(r)
        picks = []
        while len(picks) < a.n + 20:
            v = rnd.randrange(n)
            nf = r.index.meta(v).n_frames
            if nf:
                picks.append((v, rnd.randrange(nf)))
        for v, f in picks[:20]:
            r.rgb(v, f)                                   # warm the pool
        ms = []
        for v, f in picks[20:]:
            t0 = time.perf_counter()
            r.rgb(v, f)
            ms.append((time.perf_counter() - t0) * 1e3)
    ms.sort()
    p = lambda q: ms[min(len(ms) - 1, int(len(ms) * q))]   # noqa: E731
    print(f"n={len(ms)}  backend={a.backend} threads={a.threads}")
    print(f"  p50={p(.5):.2f} ms  p90={p(.9):.2f} ms  p99={p(.99):.2f} ms  "
          f"mean={sum(ms)/len(ms):.2f} ms")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="vlaframe", description=__doc__)
    ap.add_argument("--version", action="version", version=f"vlaframe {__version__}")
    sub = ap.add_subparsers(dest="cmd", required=True)

    t = sub.add_parser("transcode", help="bring videos into the target format")
    t.add_argument("src"); t.add_argument("dst")
    t.add_argument("--gop", type=int, default=30)
    t.add_argument("--crf", type=int, default=23)
    t.add_argument("--bframes", type=int, default=None,
                   help="0 removes reorder delay at about +5%% storage")
    t.add_argument("--profile", default="high")
    t.add_argument("--pix-fmt", dest="pix_fmt", default="yuv420p")
    t.add_argument("--preset", default="medium")
    t.add_argument("--max-side", dest="max_side", type=int, default=None,
                   help="cap the longer edge, preserving aspect ratio")
    t.add_argument("--threads", type=int, default=2)
    t.add_argument("--scratch", default=None)
    t.add_argument("--overwrite", action="store_true")
    t.add_argument("--limit", type=int, default=0)
    t.add_argument("--video-dirs-only", dest="video_dirs_only", action="store_true")
    t.set_defaults(fn=_cmd_transcode)

    i = sub.add_parser("index", help="build the offline .vfi index")
    i.add_argument("root"); i.add_argument("out")
    i.add_argument("--limit", type=int, default=0)
    i.add_argument("--video-dirs-only", dest="video_dirs_only", action="store_true")
    i.set_defaults(fn=_cmd_index)

    f = sub.add_parser("info", help="summarise an index")
    f.add_argument("index"); f.add_argument("--show", type=int, default=5)
    f.set_defaults(fn=_cmd_info)

    s = sub.add_parser("seal", help="mark an existing index as path-sorted, "
                                    "in place, after verifying that it is")
    s.add_argument("index", nargs="+", help="paths or globs")
    s.add_argument("--verbose", "-v", action="store_true")
    s.set_defaults(fn=_cmd_seal)

    g = sub.add_parser("get", help="decode one frame")
    g.add_argument("index"); g.add_argument("video"); g.add_argument("frame", type=int)
    g.add_argument("--backend", default="cpu", choices=("cpu", "gpu"))
    g.add_argument("--threads", type=int, default=1)
    g.add_argument("--out", default=None, help="write a PNG/JPG here")
    g.set_defaults(fn=_cmd_get)

    b = sub.add_parser("bench", help="time random single-frame reads")
    b.add_argument("index")
    b.add_argument("-n", type=int, default=300)
    b.add_argument("--backend", default="cpu", choices=("cpu", "gpu"))
    b.add_argument("--threads", type=int, default=1)
    b.add_argument("--seed", type=int, default=0)
    b.set_defaults(fn=_cmd_bench)

    a = ap.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
