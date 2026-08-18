"""Gate for `vlaframe merge`: merging must be indistinguishable from not sharding.

Sharding exists because `IndexWriter` buffers a whole index in memory, so a
corpus cannot be built in one pass. Merging is what keeps that from being
visible afterwards. The claim being tested is therefore not "the merged file
opens" or "the counts add up" -- it is that **the merged file is the file you
would have got by never sharding at all**.

That is one assertion and it is the strongest one available:

    index(A + B)  ==  merge(index(A), index(B))     byte for byte

Byte for byte, not field by field. A merge that lands one array eight bytes off
still opens, still reports the right video and frame counts, and still decodes
into plausible images for most videos -- it is the same silent class of failure
the correctness gate exists for, one layer down, and any comparison weaker than
the whole file passes it.

The rest of the checks are the cases that assertion cannot reach on its own:
parameter-set renumbering across shards, duplicate paths, empty shards, the
explicit-pos/pts flags surviving, and refusing to write over an input.

Run directly::

    python -m tests.test_merge --videos /path/to/a/directory/of/mp4s
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import os
import shutil
import sys
import tempfile

from vlaframe import (FrameReader, IndexFile, build_index, merge_indexes,
                      verify_merge)


def _sha(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _report(name: str, ok: bool, detail: str = "") -> bool:
    print(f"  {'PASS' if ok else 'FAIL'}  {name:34s} {detail}")
    return ok


# ── 1. the whole point ────────────────────────────────────────────────
def check_split_merge_identity(videos: list[str], tmp: str) -> bool:
    """index(A+B) == merge(index(A), index(B)), byte for byte.

    The split is uneven and at an arbitrary point, so the two shards differ in
    size and neither is a prefix of the whole -- a split down the middle would
    let a layout bug that happens to be symmetric survive.
    """
    whole = os.path.join(tmp, "whole.vfi")
    build_index(videos, whole)

    cut = max(1, len(videos) // 3)
    a, b = os.path.join(tmp, "a.vfi"), os.path.join(tmp, "b.vfi")
    build_index(videos[:cut], a)
    build_index(videos[cut:], b)

    merged = os.path.join(tmp, "merged.vfi")
    merge_indexes([a, b], merged, verify=True)

    same = _sha(whole) == _sha(merged)
    detail = (f"{len(videos)} videos split {cut}/{len(videos)-cut}, "
              f"{os.path.getsize(whole):,} B")
    if not same:
        detail += (f"  whole={os.path.getsize(whole)} merged="
                   f"{os.path.getsize(merged)}")
    return _report("split-merge identity", same, detail)


def check_order_independence(videos: list[str], tmp: str) -> bool:
    """The result must not depend on the order the shards are passed in.

    `merge_indexes` sorts its inputs, so this is a check that it really does --
    a merge whose output depends on argv order is one whose output depends on
    how the caller's glob happened to expand.
    """
    cut = max(1, len(videos) // 3)
    a, b = os.path.join(tmp, "a.vfi"), os.path.join(tmp, "b.vfi")
    fwd, rev = os.path.join(tmp, "fwd.vfi"), os.path.join(tmp, "rev.vfi")
    merge_indexes([a, b], fwd)
    merge_indexes([b, a], rev)
    return _report("order independence", _sha(fwd) == _sha(rev))


# ── 2. what the identity check cannot reach ───────────────────────────
def check_param_dedup(videos: list[str], tmp: str) -> bool:
    """Parameter sets are deduplicated by content across shards, and every
    video still resolves to the extradata it was built with."""
    cut = max(1, len(videos) // 3)
    a, b = os.path.join(tmp, "a.vfi"), os.path.join(tmp, "b.vfi")
    merged = os.path.join(tmp, "merged.vfi")
    fa, fb, fm = IndexFile(a), IndexFile(b), IndexFile(merged)
    try:
        distinct = {fa.extradata(i) for i in range(fa.n_params)} | \
                   {fb.extradata(i) for i in range(fb.n_params)}
        ok = fm.n_params == len(distinct)
        for src in (fa, fb):
            for vid in range(src.n_videos):
                m = src.meta(vid)
                got = fm.find(m.path)
                if got is None or fm.extradata(fm.meta(got).param_set) != \
                        src.extradata(m.param_set):
                    ok = False
                    break
        detail = (f"{fa.n_params}+{fb.n_params} in -> {fm.n_params} out "
                  f"({len(distinct)} distinct)")
    finally:
        fa.close(); fb.close(); fm.close()
    return _report("param dedup by content", ok, detail)


def check_flags_carried(videos: list[str], tmp: str) -> bool:
    """FLAG_EXPLICIT_POS/PTS survive verbatim.

    These are per-video measurements the indexer made from the packets. A merge
    has no packets, so it must carry them; recomputing would be re-deciding an
    assumption, and a wrong one pairs a frame with the wrong timestamp without
    anything looking wrong.
    """
    whole, merged = os.path.join(tmp, "whole.vfi"), os.path.join(tmp, "merged.vfi")
    fw, fm = IndexFile(whole), IndexFile(merged)
    try:
        ok = True
        both = 0
        for vid in range(fw.n_videos):
            m = fw.meta(vid)
            got = fm.find(m.path)
            if got is None or fm.meta(got).flags != m.flags:
                ok = False
                break
            both += 1
        detail = f"{both} videos, flags identical"
    finally:
        fw.close(); fm.close()
    return _report("record flags carried", ok, detail)


def check_sorted_and_readable(videos: list[str], tmp: str) -> bool:
    """The merge is path-sorted, marked as such, and actually decodes."""
    merged = os.path.join(tmp, "merged.vfi")
    r = FrameReader(merged)
    try:
        ok = r.index.paths_sorted
        prev = None
        for vid in range(r.index.n_videos):
            cur = r.index.path_bytes(vid)
            if prev is not None and cur <= prev:
                ok = False
                break
            prev = cur
        img = r.rgb(r.index.meta(0).path, 0)
        ok = ok and img.ndim == 3 and img.shape[2] == 3
        detail = f"{r.index.n_videos} videos sorted; first frame {img.shape}"
    finally:
        r.close()
    return _report("sorted, sealed, decodes", ok, detail)


def check_duplicate_policy(tmp: str) -> bool:
    a = os.path.join(tmp, "a.vfi")
    out = os.path.join(tmp, "dup.vfi")
    raised = False
    try:
        merge_indexes([a, a], out)
    except ValueError:
        raised = True
    kept = None
    if raised:
        st = merge_indexes([a, a], out, on_duplicate="first", verify=True)
        kept = st["videos"]
    src = IndexFile(a)
    expect = src.n_videos
    src.close()
    return _report("duplicate paths", raised and kept == expect,
                   f"error by default; on_duplicate='first' kept {kept}/{expect}")


def check_empty_shard(videos: list[str], tmp: str) -> bool:
    """A shard with no videos is skipped and counted, not a failure -- a
    sharded build over a directory that turned out empty is normal."""
    empty = os.path.join(tmp, "empty.vfi")
    build_index([], empty)
    a, out = os.path.join(tmp, "a.vfi"), os.path.join(tmp, "withempty.vfi")
    st = merge_indexes([a, empty], out, verify=True)
    only_a = os.path.join(tmp, "onlya.vfi")
    merge_indexes([a], only_a)
    return _report("empty shard skipped", st["empty_inputs"] == 1 and
                   _sha(out) == _sha(only_a),
                   f"{st['empty_inputs']} skipped, output identical to merge([a])")


def check_refuses_self_overwrite(tmp: str) -> bool:
    a = os.path.join(tmp, "a.vfi")
    before = _sha(a)
    raised = False
    try:
        merge_indexes([a], a)
    except ValueError:
        raised = True
    return _report("refuses output == input", raised and _sha(a) == before)


def check_failure_leaves_nothing(tmp: str) -> bool:
    """A merge that dies partway must not leave a file with a valid magic.

    Provoked with a duplicate under the default policy, which raises after the
    planning pass. The output path must not exist and no temporary must be left
    behind for a later glob to pick up.
    """
    a = os.path.join(tmp, "a.vfi")
    out = os.path.join(tmp, "never.vfi")
    try:
        merge_indexes([a, a], out)
    except ValueError:
        pass
    strays = glob.glob(os.path.join(tmp, ".merge-*"))
    return _report("failure leaves no file", not os.path.exists(out) and not strays,
                   f"{len(strays)} stray temp file(s)")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--videos", required=True,
                    help="a directory of .mp4 files, or a glob")
    ap.add_argument("-n", type=int, default=12, help="how many to use")
    ap.add_argument("--keep", action="store_true", help="keep the scratch dir")
    a = ap.parse_args(argv)

    pat = a.videos if any(c in a.videos for c in "*?[") \
        else os.path.join(a.videos, "*.mp4")
    videos = sorted(glob.glob(pat))[:a.n]
    if len(videos) < 4:
        print(f"need at least 4 videos, found {len(videos)} for {pat!r}",
              file=sys.stderr)
        return 1

    tmp = tempfile.mkdtemp(prefix="vlaframe-merge-test-")
    print(f"=== {len(videos)} videos, scratch {tmp} ===")
    try:
        results = [
            check_split_merge_identity(videos, tmp),
            check_order_independence(videos, tmp),
            check_param_dedup(videos, tmp),
            check_flags_carried(videos, tmp),
            check_sorted_and_readable(videos, tmp),
            check_duplicate_policy(tmp),
            check_empty_shard(videos, tmp),
            check_refuses_self_overwrite(tmp),
            check_failure_leaves_nothing(tmp),
        ]
    finally:
        if not a.keep:
            shutil.rmtree(tmp, ignore_errors=True)

    bad = results.count(False)
    print(f"\n{'GATE PASSED' if not bad else f'GATE FAILED: {bad} check(s)'}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
