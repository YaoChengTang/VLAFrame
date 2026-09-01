"""Bit-exactness gate for :class:`vlaframe.FrameReader`.

The failure mode this guards against is silent. A reader that returns a
neighbouring frame still returns a plausible image; the only symptom is that the
pixels stop lining up with the action/state recorded for that timestep. Every
frame-indexing bug previously found on this corpus failed exactly that way, so
nothing here is accepted on "the picture looked right".

Four things are checked, and dropping any one of them has historically produced
a false pass:

1. **Bit exactness** against a reference that opens the file with libavformat,
   seeks and decodes forward -- the path this package replaces.

2. **Keyframe boundaries specifically.** Uniformly random indices land mid-GOP
   and miss reorder bugs. The first verification round on this corpus used one
   file and twelve random indices, concluded "identical", and was wrong. Every
   keyframe position k contributes k-1, k and k+1.

3. **Both decoder lifetimes.** ``persistent`` reuses one decoder across clips;
   ``cold`` builds a fresh one per call. Five previous persistent-decoder
   implementations returned 0-51% aligned frames, and a per-file decoder was
   still 12% wrong, so a gate that tests one lifetime says nothing about the
   other.

4. **Discriminative power.** On a static shot the neighbouring frames are
   identical to the target, so returning the wrong one still passes. Each case is
   labelled by whether its neighbours actually differ, and the summary reports
   how many comparisons could have caught an off-by-one at all. A pass rate
   without that number is not evidence.

Run directly for a full sweep::

    python -m tests.test_correctness --index corpus.vfi --videos 8 --backend cpu,gpu
"""
from __future__ import annotations

import argparse
import random
import sys
from collections import Counter

import av
import numpy as np

from vlaframe import FrameReader


def reference_frame(path: str, display_frame: int, pix: str = "rgb24") -> np.ndarray:
    """Ground truth: libavformat, seek, decode forward, resolved by timestamp.

    Deliberately not "seek then take the Nth output" -- that is the bug under
    test. Resolving by timestamp is correct for any reorder depth, so a
    disagreement is the reader's.
    """
    with av.open(path) as c:
        st = c.streams.video[0]
        pts = sorted(int(pk.pts) for pk in c.demux(st)
                     if pk.pts is not None and pk.size)
    target = pts[display_frame]
    with av.open(path) as c:
        st = c.streams.video[0]
        c.seek(target, stream=st, backward=True, any_frame=False)
        for fr in c.decode(st):
            if fr.pts is not None and int(fr.pts) >= target:
                if int(fr.pts) != target:
                    raise KeyError(f"reference overshot {target} -> {fr.pts}")
                return fr.to_ndarray(format=pix)
    raise KeyError(f"reference never reached {target} in {path}")


def cases_for(reader: FrameReader, vid: int, max_kf: int, n_rand: int,
              rnd: random.Random) -> list[int]:
    """Indices to test: every keyframe boundary, plus some uniform draws."""
    m = reader.index.meta(vid)
    disp = reader.index.by_display(m)
    keys = reader.index.keys(m)
    rank = np.empty(m.n_frames, dtype=np.int64)
    rank[disp.astype(np.int64)] = np.arange(m.n_frames)
    out: set[int] = set()
    for k in keys[:max_kf]:
        base = int(rank[int(k)])
        for d in (-1, 0, 1):
            if 0 <= base + d < m.n_frames:
                out.add(base + d)
    for _ in range(n_rand):
        out.add(rnd.randrange(m.n_frames))
    return sorted(out)


def run(index_path: str, backend: str, mode: str, videos: int, max_kf: int,
        n_rand: int, seed: int, pix: str, check_power: bool,
        root: str | None = None) -> dict:
    rnd = random.Random(seed)
    r = FrameReader(index_path, root=root, backend=backend,
                    persistent=(mode == "persistent"))
    picks = rnd.sample(range(len(r)), min(videos, len(r)))
    stats: Counter = Counter()
    bad: list[tuple[str, int, str]] = []
    for vid in picks:
        m = r.index.meta(vid)
        # `m.path` is what the index *stores*, which is relative when the index
        # has a root. libavformat needs the file, so the reference opens the
        # resolved path -- passing the stored one made every reference raise,
        # and every raise is counted as an exclusion, so the gate reported a
        # pass over zero comparisons.
        src = r.index.join_root(m.path)
        for idx in cases_for(r, vid, max_kf, n_rand, rnd):
            try:
                ref = reference_frame(src, idx, pix)
            except Exception:
                stats["ref_error"] += 1
                continue
            try:
                got = r.frame(vid, idx).to_ndarray(format=pix)
            except Exception as exc:
                stats["read_error"] += 1
                bad.append((src, idx, f"{type(exc).__name__}: {exc}"))
                continue
            stats["total"] += 1
            if got.shape == ref.shape and np.array_equal(got, ref):
                stats["exact"] += 1
            else:
                d = int(np.abs(got.astype(np.int16) - ref.astype(np.int16)).max()) \
                    if got.shape == ref.shape else -1
                bad.append((src, idx, f"pixels differ, max |delta| = {d}"))
            if check_power:
                for step in (-1, 1):
                    j = idx + step
                    if not 0 <= j < m.n_frames:
                        continue
                    try:
                        other = reference_frame(src, j, pix)
                    except Exception:
                        continue
                    if other.shape == ref.shape and not np.array_equal(other, ref):
                        stats["discriminative"] += 1
                        break
    decoders = r.pool.creations
    r.close()
    return dict(stats=stats, bad=bad, decoders=decoders)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--index", required=True)
    ap.add_argument("--root", default=None,
                    help="resolve the index's stored paths against this prefix "
                         "instead of the one recorded in the file")
    ap.add_argument("--videos", type=int, default=6)
    ap.add_argument("--max-kf", dest="max_kf", type=int, default=5)
    ap.add_argument("--rand", type=int, default=5)
    ap.add_argument("--backend", default="cpu")
    ap.add_argument("--modes", default="persistent,cold")
    ap.add_argument("--pix", default="rgb24",
                    help="rgb24 goes through swscale, so a colour-conversion "
                         "difference shows up as a mismatch; yuv420p compares "
                         "the decoder's own output")
    ap.add_argument("--seed", type=int, default=20260811)
    ap.add_argument("--no-power", dest="no_power", action="store_true")
    a = ap.parse_args(argv)

    failures = 0
    for backend in [b.strip() for b in a.backend.split(",")]:
        for mode in [m.strip() for m in a.modes.split(",")]:
            res = run(a.index, backend, mode, a.videos, a.max_kf, a.rand,
                      a.seed, a.pix, not a.no_power, a.root)
            s = res["stats"]
            tot, ex = s["total"], s["exact"]
            failures += (tot - ex) + s["read_error"]
            # A run that compared nothing is not a pass. Every reference error
            # is silently excluded, so a broken reference path -- a relative
            # stored path handed to libavformat, say -- otherwise reports
            # "0/0" and GATE PASSED, which is precisely the false green this
            # file exists to refuse.
            if tot == 0:
                failures += 1
                print(f"\n=== backend={backend:4s} mode={mode:10s} pix={a.pix} ===")
                print(f"  NO COMPARISONS MADE ({s['ref_error']} reference "
                      f"error(s), {s['read_error']} read error(s)) -- a gate "
                      f"that checked nothing does not pass")
                continue
            print(f"\n=== backend={backend:4s} mode={mode:10s} pix={a.pix} ===")
            print(f"  cases            : {tot}")
            print(f"  bit-exact        : {ex}/{tot}"
                  f"{'   ALL PASS' if tot and ex == tot else ''}")
            if not a.no_power:
                print(f"  discriminative   : {s['discriminative']}/{tot} "
                      f"({100*s['discriminative']/max(tot,1):.0f}% could have "
                      f"caught an off-by-one)")
            print(f"  decoders created : {res['decoders']}")
            if s["read_error"]:
                print(f"  read errors      : {s['read_error']}")
            if s["ref_error"]:
                print(f"  reference errors : {s['ref_error']} (excluded)")
            for p, i, why in res["bad"][:6]:
                print(f"    MISMATCH frame {i}: {why}  {p.split('/')[-1][:48]}")
    print(f"\n{'GATE PASSED' if failures == 0 else f'GATE FAILED: {failures} bad case(s)'}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
