"""Command line: ``vlaframe transcode | index | merge | rebase | info | seal |
get | bench``.

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
from .merge import merge_indexes
from .rebase import rebase_index
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
    # Storing paths relative to the tree being walked is the default: an index
    # whose strings carry the mount point stops matching the moment the corpus
    # is mounted anywhere else, and nothing about that fails at open time.
    store_root = "" if a.absolute or root.is_file() else str(root)
    print(f"indexing {len(files):,} file(s) under {root}"
          f"{'' if store_root else ' (absolute paths)'}", flush=True)
    t0 = time.perf_counter()

    def progress(i, n, path):
        if i % 500 == 0:
            print(f"  [{i}/{n}] {time.perf_counter()-t0:.0f}s", flush=True)

    st = build_index(files, a.out, root=store_root, progress=progress)
    print(f"\nindexed {st['videos']:,} videos, {st['frames']:,} frames, "
          f"{st['param_sets']} parameter set(s), {st['failed']} failed")
    print(f"  root: {st['root'] or '(none, paths are absolute)'}")
    print(f"index {st['bytes']/1e6:.1f} MB = {st['bytes_per_frame']:.2f} B/frame; "
          f"{st['explicit_pos']} video(s) needed explicit byte offsets; "
          f"built in {time.perf_counter()-t0:.0f}s")
    for p, e in st["failures"]:
        print(f"  FAILED {e}  {p}")
    return 0


def _cmd_merge(a) -> int:
    """Combine index shards into one `.vfi`.

    Shards are how a corpus-scale index gets *built* (one `IndexWriter` cannot
    buffer seven million videos); merging is what keeps that from leaking into
    how it is *read*.
    """
    inputs = []
    for p in a.inputs:
        inputs.extend(sorted(glob.glob(p)) if any(c in p for c in "*?[") else [p])
    if not inputs:
        print("no input indexes matched", file=sys.stderr)
        return 1
    print(f"merging {len(inputs)} shard(s) -> {a.out}", flush=True)
    t0 = time.perf_counter()

    def progress(i, n, path):
        if i % 20 == 0 or i == n:
            print(f"  planned [{i}/{n}] {time.perf_counter()-t0:.0f}s", flush=True)

    st = merge_indexes(inputs, a.out, on_duplicate=a.on_duplicate,
                       verify=a.verify, progress=progress)
    print(f"\n{st['videos']:,} videos, {st['frames']:,} frames, "
          f"{st['input_param_sets']} -> {st['param_sets']} parameter set(s)")
    if st["duplicates"]:
        print(f"  {st['duplicates']} duplicate path(s) resolved by "
              f"--on-duplicate={a.on_duplicate}")
    if st["empty_inputs"]:
        print(f"  {st['empty_inputs']} empty shard(s) skipped")
    print(f"  {st['explicit_pos']} explicit pos, {st['explicit_pts']} explicit pts "
          f"(carried, never recomputed)")
    print(f"  {st['bytes']/1e6:.1f} MB = {st['bytes_per_frame']:.2f} B/frame "
          f"in {time.perf_counter()-t0:.0f}s")
    if a.verify:
        v = st["verified"]
        print(f"  VERIFIED {v['videos']:,} videos / {v['arrays']:,} arrays "
              f"byte-for-byte against {v['inputs']} input(s)")
    return 0


def _cmd_info(a) -> int:
    with FrameReader(a.index, root=a.root) as r:
        n = len(r)
        frames = sum(r.index.meta(i).n_frames for i in range(n))
        print(f"{a.index}")
        print(f"  videos       : {n:,}")
        print(f"  frames       : {frames:,}")
        print(f"  parameter sets: {r.index.n_params}")
        lookup = "bisect (paths sorted)" if r.index.paths_sorted else \
            "dict (paths not sorted -- see `vlaframe seal`)"
        print(f"  path lookup  : {lookup}")
        print(f"  paths        : "
              + ("relative to a root" if r.index.relative_paths else "absolute"))
        print(f"  stored root  : {r.index.stored_root or '(none)'}")
        if r.index.root != r.index.stored_root:
            print(f"  root override: {r.index.root}")
        print(f"  file size    : {os.path.getsize(a.index)/1e6:.1f} MB "
              f"({os.path.getsize(a.index)/max(frames,1):.2f} B/frame)")
        for i in range(min(a.show, n)):
            m = r.index.meta(i)
            print(f"    [{i}] {m.width}x{m.height} {m.n_frames:>7} frames  "
                  f"{m.n_keys:>5} keys  {m.path}")
    return 0


def _cmd_rebase(a) -> int:
    """Move an index's stored paths onto a different root.

    The paths are ~1.5% of a `.vfi` and the per-frame arrays are the rest, and
    nothing in the arrays depends on the path -- so changing where a corpus
    lives is a rewrite of that 1.5%, not a rebuild.
    """
    paths = []
    for p in a.index:
        paths.extend(sorted(glob.glob(p)) if any(c in p for c in "*?[") else [p])
    if not paths:
        print("no index files matched", file=sys.stderr)
        return 1
    if a.out and len(paths) > 1:
        print("--out takes one index; pass --in-place for a sweep",
              file=sys.stderr)
        return 1

    done = failed = 0
    for p in paths:
        try:
            st = rebase_index(p, root=a.root, to_absolute=a.to_absolute,
                              out_path=a.out, in_place=a.in_place)
        except Exception as exc:
            failed += 1
            print(f"  FAILED {p}\n    {type(exc).__name__}: {exc}", flush=True)
            continue
        done += 1
        where = "in place" if st["in_place"] else f"-> {st['out']}"
        print(f"  {st['videos']:>8,} videos  {where}", flush=True)
        print(f"      root  {st['old_root'] or '(none)'} -> "
              f"{st['new_root'] or '(none)'}", flush=True)
        print(f"      first {st['sample']}", flush=True)
        print(f"      strings {st['strings_bytes']:,} B, "
              f"{st['gap_bytes']:,} B freed before the records", flush=True)
    print(f"\nrebased {done}/{len(paths)}"
          f"{f', {failed} failed' if failed else ''}")
    return 1 if failed else 0


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
    with FrameReader(a.index, root=a.root, backend=a.backend,
                     threads=a.threads) as r:
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
    with FrameReader(a.index, root=a.root, backend=a.backend,
                     threads=a.threads) as r:
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
    i.add_argument("--absolute", action="store_true",
                   help="store absolute paths instead of paths relative to "
                        "<root>. The index then only matches this mount point.")
    i.set_defaults(fn=_cmd_index)

    m = sub.add_parser("merge", help="combine index shards into one .vfi")
    m.add_argument("out")
    m.add_argument("inputs", nargs="+", help="paths or globs")
    m.add_argument("--on-duplicate", dest="on_duplicate", default="error",
                   choices=("error", "first", "last"))
    m.add_argument("--verify", action="store_true",
                   help="re-open output and inputs and compare every video and "
                        "every array byte for byte")
    m.set_defaults(fn=_cmd_merge)

    f = sub.add_parser("info", help="summarise an index")
    f.add_argument("index"); f.add_argument("--show", type=int, default=5)
    f.add_argument("--root", default=None,
                   help="resolve stored paths against this prefix instead of "
                        "the one recorded in the file")
    f.set_defaults(fn=_cmd_info)

    rb = sub.add_parser("rebase", help="move an index's stored paths onto a "
                                       "different root, without rebuilding")
    rb.add_argument("index", nargs="+", help="paths or globs")
    rb.add_argument("--root", default=None,
                    help="the prefix the stored paths become relative to")
    rb.add_argument("--to-absolute", dest="to_absolute", action="store_true",
                    help="inverse: fold the root back into every path")
    rb.add_argument("--out", default=None,
                    help="write here instead of <index>.rebased (one index only)")
    rb.add_argument("--in-place", dest="in_place", action="store_true",
                    help="edit the input. Cheap (the arrays are never read) and "
                         "destructive; without it the input is left untouched.")
    rb.set_defaults(fn=_cmd_rebase)

    s = sub.add_parser("seal", help="mark an existing index as path-sorted, "
                                    "in place, after verifying that it is")
    s.add_argument("index", nargs="+", help="paths or globs")
    s.add_argument("--verbose", "-v", action="store_true")
    s.set_defaults(fn=_cmd_seal)

    g = sub.add_parser("get", help="decode one frame")
    g.add_argument("index"); g.add_argument("video"); g.add_argument("frame", type=int)
    g.add_argument("--backend", default="cpu", choices=("cpu", "gpu"))
    g.add_argument("--threads", type=int, default=1)
    g.add_argument("--root", default=None,
                   help="resolve stored paths against this prefix")
    g.add_argument("--out", default=None, help="write a PNG/JPG here")
    g.set_defaults(fn=_cmd_get)

    b = sub.add_parser("bench", help="time random single-frame reads")
    b.add_argument("index")
    b.add_argument("-n", type=int, default=300)
    b.add_argument("--backend", default="cpu", choices=("cpu", "gpu"))
    b.add_argument("--threads", type=int, default=1)
    b.add_argument("--root", default=None,
                   help="resolve stored paths against this prefix")
    b.add_argument("--seed", type=int, default=0)
    b.set_defaults(fn=_cmd_bench)

    a = ap.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
