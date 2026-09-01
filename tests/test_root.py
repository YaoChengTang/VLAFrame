"""Gate for the index root: what a `.vfi` asserts, and what it does not.

An index used to store absolute paths, and `IndexFile.find` matches by exact
bytes. That made a `.vfi` a function of where the corpus was mounted and of what
the directories above the videos were called -- rename a dataset directory and
every lookup misses, with nothing failing when the index opens. What the index
should assert is the shape *below* a root; where that root is, is the caller's
business.

Five things are checked, and each one is here because getting it wrong is quiet
rather than loud:

1. **A relative index decodes what an absolute one decodes.** Same videos, same
   frames, byte for byte. If the two disagree the stored strings are addressing
   something other than what they name.

2. **A root override relocates an index without rebuilding it.** This is the
   whole point: a corpus that moved needs a caller that says so.

3. **An override is ignored on an absolute index.** This is the regression that
   prompted the file. A consumer passes `root=` for every leaf it opens, and
   some of those indexes are not migrated yet. Honouring the override there
   strips the prefix off queries and compares the remainder against absolute
   records, so *every* lookup misses -- turning "your old indexes keep working"
   into "all of them broke at once", at read time, one frame at a time.

4. **`find` does not normalise.** It is a byte comparison, so a symlinked path
   and its realpath are different keys. Worth stating as a property rather than
   discovering it: it is why the root is a separate field and not something
   recovered by string surgery at lookup time.

5. **rebase round-trips.** Absolute -> relative -> absolute must reproduce the
   original file byte for byte, or the migration is lossy in some way nobody
   will notice until a lookup misses.

Run directly::

    python -m tests.test_root --videos /path/to/a/directory/of/mp4s
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
                      rebase_index)


def _sha(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _report(name: str, ok: bool, detail: str = "") -> bool:
    print(f"  {'PASS' if ok else 'FAIL'}  {name:38s} {detail}")
    return ok


def _frames(index: str, root: str | None = None) -> list[str]:
    """Hash a fixed set of frames of every video, addressed by id."""
    out = []
    with FrameReader(index, root=root) as r:
        for vid in range(len(r)):
            n = r.index.meta(vid).n_frames
            for fr in (0, n // 2, n - 1):
                out.append(hashlib.sha256(r.rgb(vid, fr).tobytes()).hexdigest())
    return out


def check_relative_matches_absolute(videos, tmp) -> bool:
    src = os.path.dirname(videos[0])
    a = os.path.join(tmp, "abs.vfi")
    b = os.path.join(tmp, "rel.vfi")
    build_index(videos, a)
    build_index(videos, b, root=src)
    with IndexFile(b) as f:
        stored = [f.meta(i).path for i in range(f.n_videos)]
        rel = all(not os.path.isabs(p) for p in stored) and f.stored_root == src
    ok = _report("stored paths are relative", rel, stored[0] if stored else "")
    ok &= _report("relative decodes as absolute", _frames(a) == _frames(b),
                  f"{len(videos) * 3} frames")
    # both spellings of the same video resolve
    with IndexFile(b) as f:
        ok &= _report("find takes relative and absolute",
                      f.find(stored[0]) == 0
                      and f.find(os.path.join(src, stored[0])) == 0)
        ok &= _report("find rejects a path outside the root",
                      f.find("/nowhere/x.mp4") is None)
    return ok


def check_override_relocates(videos, tmp) -> bool:
    src = os.path.dirname(videos[0])
    rel = os.path.join(tmp, "rel2.vfi")
    build_index(videos, rel, root=src)
    before = _frames(rel)

    moved = os.path.join(tmp, "corpus_renamed")
    shutil.copytree(src, moved)
    ok = _report("override relocates without a rebuild",
                 _frames(rel, root=moved) == before, os.path.basename(moved))
    with IndexFile(rel, root=moved) as f:
        ok &= _report("override does not touch the file",
                      f.stored_root == src and f.root == moved)
        ok &= _report("find hits under the overridden root",
                      f.find(os.path.join(moved, f.meta(0).path)) == 0)
    return ok


def check_override_ignored_when_absolute(videos, tmp) -> bool:
    """The regression: `root=` on an index that stores absolute paths.

    A consumer passes it for every leaf, and an unmigrated index has no prefix
    to replace. Stripping one off the query anyway makes every lookup miss.
    """
    src = os.path.dirname(videos[0])
    a = os.path.join(tmp, "abs2.vfi")
    build_index(videos, a)                       # absolute, as before roots
    expect = _frames(a)

    with IndexFile(a, root=src) as f:
        ok = _report("override flagged as ignored", f.root_override_ignored)
        ok &= _report("root stays empty on an absolute index", f.root == "")
        ok &= _report("find still hits with an override passed",
                      f.find(f.meta(0).path) == 0)
    ok &= _report("absolute index decodes under an override",
                  _frames(a, root=src) == expect, f"{len(expect)} frames")
    # and a wrong override must not silently start resolving either
    with IndexFile(a, root="/somewhere/else") as f:
        ok &= _report("a wrong override changes nothing",
                      f.find(f.meta(0).path) == 0)
    return ok


def check_find_is_bytes(videos, tmp) -> bool:
    """`find` compares bytes; a link and its target are different keys."""
    src = os.path.dirname(videos[0])
    rel = os.path.join(tmp, "rel3.vfi")
    build_index(videos, rel, root=src)
    with IndexFile(rel) as f:
        stem = f.meta(0).path
        ok = _report("trailing-slash root normalises",
                     IndexFile(rel, root=src + "/").find(
                         os.path.join(src, stem)) == 0)
        ok &= _report("a near-miss path does not resolve",
                      f.find(stem + "x") is None and f.find("./" + stem) is None)
    return ok


def check_rebase_round_trip(videos, tmp) -> bool:
    src = os.path.dirname(videos[0])
    a = os.path.join(tmp, "rt.vfi")
    build_index(videos, a)
    sha0, expect = _sha(a), _frames(a)

    fwd = rebase_index(a, root=src, out_path=os.path.join(tmp, "rt_rel.vfi"))
    ok = _report("rebase leaves the input alone", _sha(a) == sha0)
    with IndexFile(fwd["out"]) as f:
        ok &= _report("rebased index is relative",
                      f.relative_paths and not os.path.isabs(f.meta(0).path))
    ok &= _report("rebased index decodes identically",
                  _frames(fwd["out"]) == expect)

    back = rebase_index(fwd["out"], to_absolute=True,
                        out_path=os.path.join(tmp, "rt_abs.vfi"))
    ok &= _report("round-trip is byte-identical", _sha(back["out"]) == sha0,
                  f"{sha0[:12]} vs {_sha(back['out'])[:12]}")

    ip = os.path.join(tmp, "ip.vfi")
    shutil.copyfile(a, ip)
    rebase_index(ip, root=src, in_place=True)
    ok &= _report("--in-place matches the copying form",
                  _sha(ip) == _sha(fwd["out"]) and _frames(ip) == expect)
    return ok


def check_merge_refuses_mixed_roots(videos, tmp) -> bool:
    """Shards relative to different places cannot be one index."""
    src = os.path.dirname(videos[0])
    half = len(videos) // 2
    a = os.path.join(tmp, "m_rel.vfi")
    b = os.path.join(tmp, "m_abs.vfi")
    build_index(videos[:half], a, root=src)
    build_index(videos[half:], b)                # no root
    try:
        merge_indexes([a, b], os.path.join(tmp, "m_out.vfi"))
    except ValueError as exc:
        return _report("merge refuses mixed roots", "root" in str(exc).lower(),
                       str(exc).splitlines()[0][:52])
    return _report("merge refuses mixed roots", False, "it did not refuse")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--videos", required=True,
                    help="a directory of mp4s, or a glob")
    ap.add_argument("-n", type=int, default=6)
    ap.add_argument("--keep", action="store_true")
    a = ap.parse_args(argv)

    pat = a.videos if any(c in a.videos for c in "*?[") \
        else os.path.join(a.videos, "*.mp4")
    videos = sorted(glob.glob(pat))[:a.n]
    if len(videos) < 2:
        print(f"need at least 2 videos, found {len(videos)} for {pat!r}",
              file=sys.stderr)
        return 1

    tmp = tempfile.mkdtemp(prefix="vlaframe-root-test-")
    print(f"=== {len(videos)} videos, scratch {tmp} ===")
    try:
        results = [
            check_relative_matches_absolute(videos, tmp),
            check_override_relocates(videos, tmp),
            check_override_ignored_when_absolute(videos, tmp),
            check_find_is_bytes(videos, tmp),
            check_rebase_round_trip(videos, tmp),
            check_merge_refuses_mixed_roots(videos, tmp),
        ]
    finally:
        if not a.keep:
            shutil.rmtree(tmp, ignore_errors=True)

    bad = results.count(False)
    print(f"\n{'GATE PASSED' if not bad else f'GATE FAILED: {bad} check(s)'}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
