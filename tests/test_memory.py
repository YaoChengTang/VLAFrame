"""Gate for the R8 change: the reader must hold no per-video state.

The bug this guards against does not crash and does not look wrong. It is a
worker whose resident memory climbs with the number of videos it has touched,
so a run survives the smoke test, survives the first thousand steps, and then
dies to the OOM killer somewhere in the middle of an epoch -- by which point
the evidence is gone. On the corpus this was written for (6.98 M videos,
2.97 B frames) the caches that were removed measured 27.8 GB per worker, and a
training job runs 32 of them.

Four things are checked, and the first two exist because the third is easy to
pass by accident:

1. **Bisection agrees with the dict, for every video in the index.** Not a
   sample. A bisection that lands one record off returns another video's
   frames, which decodes to a perfectly plausible image; only an exhaustive
   comparison rules it out.

2. **A path the index does not cover resolves to None**, and the neighbours of
   a real path (one byte shorter, one byte longer, prefix, suffix) do not
   collide with it. Off-by-one at the boundary is the other way a bisection
   silently aliases.

3. **Private anonymous memory is flat** across a long run of reads that touches
   every video. Anonymous, because the file-backed pages of the mmap-ed index
   are *meant* to grow. Judged on net growth after a collection, not on a
   fitted slope: at a ~16 MB heap the fit is dominated by allocator churn and
   came out positive on a run whose anonymous memory had fallen 2 MB.

4. **No file descriptor is retained** between calls, and none is stranded when
   a read raises. Not merely bounded: the reader keeps none at all. A cache
   would not earn its place here -- a run that cannot finish an epoch touches
   each video about once, so the hit rate is structurally zero, and the
   ``os.open`` it would save costs 3.1 ms on this corpus's NFS mount either way
   (that export sets ``lookupcache=none``). That tax is fixed at the mount.

Run directly::

    python -m tests.test_memory --index corpus.vfi
    python -m tests.test_memory --index corpus.vfi --reads 100000   # slow gate
"""
from __future__ import annotations

import argparse
import gc
import os
import random
import sys

from vlaframe import FrameReader, IndexFile


def rss_anon_bytes() -> int:
    """Private anonymous resident memory of this process.

    ``RssAnon``, not total RSS. The caches being guarded against -- dicts of
    unpacked records, numpy permutations, a path -> id map -- are Python heap,
    which is anonymous. Total RSS also counts the file-backed pages of the
    mmap-ed index, and those *are* meant to grow: faulting in the records of the
    videos actually sampled is the design. A gate on total RSS fails on a large
    index for the intended behaviour (measured +20 MB over 800 reads on a 203 MB
    index, exactly the ~20 KB/video of index pages touched) and would have to be
    loosened until it stopped catching anything.
    """
    with open("/proc/self/status") as fh:
        for line in fh:
            if line.startswith("RssAnon:"):
                return int(line.split()[1]) * 1024
    raise RuntimeError("RssAnon not reported by this kernel")


def rss_file_bytes() -> int:
    """File-backed resident memory: the mmap-ed index pages, reported for
    context so a run shows what it faulted in as well as what it kept."""
    with open("/proc/self/status") as fh:
        for line in fh:
            if line.startswith("RssFile:"):
                return int(line.split()[1]) * 1024
    return 0


def open_fds() -> int:
    return len(os.listdir("/proc/self/fd"))


# ── 1. bisection vs dict, exhaustively ────────────────────────────────
def check_lookup_equivalence(index_path: str) -> dict:
    f = IndexFile(index_path)
    if not f.paths_sorted:
        f.close()
        return dict(name="lookup equivalence", skipped=True,
                    detail="index is not sealed; `find` is the dict path itself")
    # The dict is what the bisection replaces, so building it here is the
    # point -- it is the reference, not something the reader would do.
    ref = f.path_to_id()
    bad = []
    for path, vid in ref.items():
        got = f.find(path)
        if got != vid:
            bad.append((path, vid, got))
            if len(bad) >= 5:
                break
    n = f.n_videos
    f.close()
    return dict(name="lookup equivalence", checked=n, bad=bad,
                detail=f"{n - len(bad)}/{n} videos resolve identically")


# ── 2. misses and near-misses ─────────────────────────────────────────
def check_lookup_misses(index_path: str, rnd: random.Random) -> dict:
    f = IndexFile(index_path)
    bad = []
    if f.n_videos:
        for _ in range(min(64, f.n_videos)):
            vid = rnd.randrange(f.n_videos)
            real = f.meta(vid).path
            if f.find(real) != vid:
                bad.append(("exact", real, vid, f.find(real)))
            # Neighbours in byte order that must NOT resolve to `real`.
            for label, probe in (("truncated", real[:-1]),
                                 ("extended", real + "x"),
                                 ("prefix-drop", real[1:]),
                                 ("suffixed", real + "/")):
                got = f.find(probe)
                if got is not None and f.meta(got).path != probe:
                    bad.append((label, probe, None, got))
    for absent in ("", "/", "/nonexistent/video.mp4", "\x00", "￿"):
        got = f.find(absent)
        if got is not None:
            bad.append(("absent", absent, None, got))
    n = f.n_videos
    f.close()
    return dict(name="lookup misses", checked=min(64, n) * 5 + 5, bad=bad,
                detail="absent and near-miss paths all resolve to None")


# ── 3. resident memory does not grow with videos touched ──────────────
def check_rss_flat(index_path: str, reads: int, rnd: random.Random,
                   warmup: int = 200) -> dict:
    """Read `reads` random frames and fit RSS growth per read.

    Videos are drawn to cover the index rather than uniformly at random: the
    failure mode is memory proportional to *distinct* videos touched, and
    uniform draws over a large index repeat too rarely to distinguish "grows
    per video" from "grows per read" anyway. Sweeping in order of video id
    guarantees the worst case for the removed caches.
    """
    r = FrameReader(index_path)
    n = len(r)
    picks = []
    for i in range(reads + warmup):
        # Sweep in video-id order rather than drawing uniformly: the failure
        # mode is memory proportional to *distinct* videos touched, so covering
        # the index is the point. Uniform draws over a large index repeat too
        # rarely to tell "grows per video" from "grows per read" anyway.
        vid = i % n
        nf = r.index.meta(vid).n_frames
        if nf:
            picks.append((vid, rnd.randrange(nf)))

    errors = 0
    for vid, fr in picks[:warmup]:                     # let the pool settle
        try:
            r.rgb(vid, fr)
        except Exception:
            errors += 1

    samples = []
    every = max(1, len(picks[warmup:]) // 40)
    file0 = rss_file_bytes()
    gc.collect()
    for i, (vid, fr) in enumerate(picks[warmup:]):
        try:
            r.rgb(vid, fr)
        except Exception:
            errors += 1
        if i % every == 0:
            # Collect before sampling: a cycle that has not been reaped yet is
            # noise, and at this heap size (~16 MB) allocator churn between
            # samples is larger than the growth a bounded run would show.
            gc.collect()
            samples.append((i, rss_anon_bytes()))
    gc.collect()
    samples.append((len(picks) - warmup, rss_anon_bytes()))
    file1 = rss_file_bytes()
    r.close()

    # Net growth, not a least-squares slope. The slope is reported for context
    # but is not the gate: over 40 noisy samples of a 16 MB heap it came out
    # positive on a run whose anonymous memory had *fallen* 2 MB end to end.
    # What an unbounded per-video memo produces is monotone net growth --
    # 11.6 KB per distinct video on this corpus (8 B/frame for the permutation
    # plus 581 B for the record and path) -- which dwarfs allocator noise.
    n_reads = len(picks) - warmup
    net = samples[-1][1] - samples[0][1]
    k = len(samples)
    mx = sum(s[0] for s in samples) / k
    my = sum(s[1] for s in samples) / k
    den = sum((s[0] - mx) ** 2 for s in samples)
    slope = sum((s[0] - mx) * (s[1] - my) for s in samples) / den if den else 0.0
    return dict(name="anon rss flat", reads=n_reads, errors=errors,
                net_per_read=net / max(n_reads, 1), slope=slope,
                detail=f"net {net/1e6:+.1f} MB = {net/max(n_reads,1):+.0f} B/read "
                       f"over {n_reads:,} reads "
                       f"(RssAnon {samples[0][1]/1e6:.0f} -> {samples[-1][1]/1e6:.0f} MB, "
                       f"fitted slope {slope:+.0f} B/read; "
                       f"RssFile {file0/1e6:.0f} -> {file1/1e6:.0f} MB, "
                       f"index pages, expected to grow)")


# ── 4. no descriptor is retained or leaked ────────────────────────────
def check_no_fd_growth(index_path: str, rnd: random.Random,
                       reads: int = 200) -> dict:
    """Descriptors must not accumulate at all, and none stranded on failure.

    Strict rather than bounded: the reader keeps no descriptor between calls, so
    every open is matched by a close and the count must not move. The version
    this replaced reached one descriptor per video touched.
    """
    r = FrameReader(index_path)
    n = len(r)

    def read_random():
        vid = rnd.randrange(n)
        nf = r.index.meta(vid).n_frames
        if nf:
            r.rgb(vid, rnd.randrange(nf))

    for _ in range(20):
        read_random()
    before = open_fds()
    for _ in range(reads):
        read_random()
    after = open_fds()

    # The descriptor is opened and closed inside one `frame()` call, so the only
    # window in which it can be stranded is a raise between the two. Injecting
    # the fault is the only way to exercise that window -- an out-of-range frame
    # raises in `_plan` before the open, and a missing file raises at the open
    # itself, so neither would notice a missing `finally`.
    import vlaframe.reader as _reader
    real_pread = _reader.os.pread
    _reader.os.pread = lambda *a, **k: (_ for _ in ()).throw(OSError("injected"))
    try:
        for _ in range(50):
            try:
                read_random()
            except OSError:
                pass
    finally:
        _reader.os.pread = real_pread
    after_bad = open_fds()
    r.close()

    bad = []
    if after > before:
        bad.append(("accumulated across reads", before, after))
    if after_bad > before:
        bad.append(("stranded on failed read", before, after_bad))
    return dict(name="fd stable", bad=bad,
                detail=f"{before} -> {after} over {reads} reads, "
                       f"{after_bad} after 50 injected failures")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--index", required=True)
    ap.add_argument("--reads", type=int, default=4000,
                    help="frames to read for the RSS slope; the real gate is "
                         "a full sweep of the index")
    ap.add_argument("--growth-limit", dest="slope_limit", type=float, default=1024.0,
                    help="bytes of net RssAnon growth per read that still counts "
                         "as flat. The caches removed produced 11.6 KB per "
                         "distinct video on this corpus, so the gate has an "
                         "order of magnitude of headroom over allocator noise")
    ap.add_argument("--seed", type=int, default=20260818)
    ap.add_argument("--skip-equivalence", dest="skip_equiv", action="store_true",
                    help="the exhaustive lookup check builds the dict it is "
                         "testing against, which needs the memory that dict costs")
    a = ap.parse_args(argv)

    rnd = random.Random(a.seed)
    results = []
    if not a.skip_equiv:
        results.append(check_lookup_equivalence(a.index))
    results.append(check_lookup_misses(a.index, rnd))
    results.append(check_rss_flat(a.index, a.reads, rnd))
    results.append(check_no_fd_growth(a.index, rnd))

    failures = 0
    print(f"=== {a.index} ===")
    for res in results:
        if res.get("skipped"):
            print(f"  SKIP  {res['name']:22s} {res['detail']}")
            continue
        bad = res.get("bad", [])
        if res["name"] == "anon rss flat":
            ok = res["net_per_read"] <= a.slope_limit and res["errors"] == 0
        else:
            ok = not bad
        failures += 0 if ok else 1
        print(f"  {'PASS' if ok else 'FAIL'}  {res['name']:22s} {res['detail']}")
        for row in bad[:5]:
            print(f"          {row}")
        if res["name"] == "anon rss flat" and res["errors"]:
            print(f"          {res['errors']} read error(s)")

    print(f"\n{'GATE PASSED' if failures == 0 else f'GATE FAILED: {failures} check(s)'}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
