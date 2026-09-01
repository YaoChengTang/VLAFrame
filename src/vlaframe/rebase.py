"""Function 5 — move an existing index onto a different path root.

An index built before roots existed stores each video's absolute path, so it is
a function of where the corpus was mounted and of what the directories above the
videos were called. Rebuilding one to change that is absurdly expensive: the
paths are ~1.5% of the file and the per-frame arrays are the other 98.5%, and
recomputing the arrays means demuxing every mp4 again -- 480,000 files and 2.35
TB on the corpus this was written for.

Nothing about the arrays depends on the path. So this rewrites only the parts
that do:

    header      off_root, and nothing else
    root        the new prefix
    strings     the paths, with the prefix added or removed
    records     each (path_off, path_len) pair

and leaves every byte from ``off_arrays`` onward exactly where it was. Removing
a common prefix shortens the strings section, so the freed bytes become a gap
before ``off_records``; the gap is zero-filled, which is what ``_pad_to`` had
already put there, so a rebase and its inverse round-trip to the identical file.

Why the order survives
----------------------
``HFLAG_PATHS_SORTED`` says the records are ordered by their stored bytes, and
``IndexFile.find`` bisects on that. Adding or removing one prefix that *every*
path shares is order-preserving, so the flag stays true -- but "every path
shares it" is checked here rather than assumed, because a flag that lies makes
the bisection return **another video's** frames, which decode into a perfectly
plausible image.

Growing does not always fit
---------------------------
Going the other way (relative -> absolute) lengthens the strings. An index that
was *built* relative has no gap to grow into, and moving ``off_records`` would
mean relocating the arrays, i.e. the rewrite this module exists to avoid. That
case is refused with the shortfall named rather than silently rebuilt.
"""
from __future__ import annotations

import os
import shutil
import struct
import tempfile

from ._format import (HEADER_FMT, PATH_FMT, RECORD_SIZE, IndexFile,
                      normalize_root, relative_to_root)


def _plan_paths(f: IndexFile, new_root: str) -> list[str]:
    """The stored string each record should hold under `new_root`.

    Raises when a video does not live under the new root -- the whole point of
    a root is that it is common, and a rebase that quietly dropped or mangled
    the odd path would leave an index that looks complete and is not.
    """
    out = []
    for vid in range(f.n_videos):
        absolute = f.join_root(f.meta(vid).path)
        if not new_root:
            out.append(absolute)
            continue
        rel = relative_to_root(absolute, new_root)
        if rel is None:
            raise ValueError(
                f"{absolute} is not under the requested root {new_root!r}. "
                f"Every video in one index has to share its root.")
        out.append(rel)
    return out


def _rewrite(target: str, old_root: str, new_root: str,
             new_paths: list[str]) -> dict:
    """Apply the plan to `target` in place. Caller has already validated it."""
    with open(target, "r+b") as fh:
        head = fh.read(struct.calcsize(HEADER_FMT))
        (magic, version, n_videos, n_params, flags, off_paramtab, off_strings,
         off_records, off_arrays, _old_root_off) = struct.unpack_from(
            HEADER_FMT, head, 0)

        # The root sits between the param blobs and the strings, so the fixed
        # point is the END OF THE BLOBS, not the current start of the strings.
        # Anchoring on off_strings instead makes the strings drift right by one
        # root every time the file is rebased, and the inverse rebase then has
        # nowhere to put the longer paths.
        old_root_blob = (old_root.encode() + b"\0") if old_root else b""
        anchor = off_strings - len(old_root_blob)

        root_blob = (new_root.encode() + b"\0") if new_root else b""
        off_root = anchor if root_blob else 0
        new_off_strings = anchor + len(root_blob)
        strings = b"".join(p.encode() + b"\0" for p in new_paths)
        end = new_off_strings + len(strings)
        if end > off_records:
            raise ValueError(
                f"{target}: the rebased paths need {end - off_records} more "
                f"byte(s) than the space before the records. Growing the "
                f"strings would move the arrays, which is a rebuild, not a "
                f"rebase.")

        fh.seek(0)
        fh.write(struct.pack(HEADER_FMT, magic, version, n_videos, n_params,
                             flags, off_paramtab, new_off_strings, off_records,
                             off_arrays, off_root))
        fh.seek(anchor)
        fh.write(root_blob)
        fh.write(strings)
        # Zero the bytes the shorter strings freed. `_pad_to` had already put
        # zeros here when the file was written, so this is what makes a rebase
        # and its inverse produce the identical file rather than one carrying
        # the ghost of the longer paths.
        fh.write(b"\0" * (off_records - end))

        offset = 0
        for vid, p in enumerate(new_paths):
            fh.seek(off_records + RECORD_SIZE * vid)
            fh.write(struct.pack(PATH_FMT, offset, len(p.encode())))
            offset += len(p.encode()) + 1
        fh.flush()
        os.fsync(fh.fileno())
    return dict(strings_bytes=len(strings), gap_bytes=off_records - end)


def rebase_index(path: str, *, root: str | None = None,
                 to_absolute: bool = False, out_path: str | None = None,
                 in_place: bool = False) -> dict:
    """Rewrite `path`'s stored paths onto `root` (or make them absolute).

    Writes a new file by default -- ``out_path``, else ``<path>.rebased`` --
    and leaves the input untouched, so a bad root costs a delete rather than an
    index. ``in_place=True`` edits the original; it is the cheap form (the
    arrays are never even read) and the destructive one, so it is opt-in.

    ``to_absolute=True`` is the inverse: it folds the root back into every path
    and clears it, which is how a rebase is undone.
    """
    if to_absolute and root:
        raise ValueError("pass either root= or to_absolute=True, not both")
    if in_place and out_path:
        raise ValueError("in_place=True writes the input; out_path is meaningless")
    if not to_absolute and root is None:
        raise ValueError("rebase needs root= (or to_absolute=True)")

    new_root = "" if to_absolute else normalize_root(root)

    with IndexFile(path) as f:
        old_root = f.stored_root
        new_paths = _plan_paths(f, new_root)
        sorted_before = f.paths_sorted
        n_videos = f.n_videos

    # A prefix shared by every path is order-preserving, but the flag is what
    # `find` trusts, so verify rather than reason about it.
    if sorted_before:
        encoded = [p.encode() for p in new_paths]
        for i in range(1, len(encoded)):
            if encoded[i] <= encoded[i - 1]:
                raise ValueError(
                    f"{path}: rebasing onto {new_root!r} breaks path order at "
                    f"record {i} ({encoded[i-1]!r} then {encoded[i]!r}), and the "
                    f"index is marked sorted. Rebuild it instead.")

    if in_place:
        target = path
        detail = _rewrite(target, old_root, new_root, new_paths)
    else:
        target = out_path or (path + ".rebased")
        out_dir = os.path.dirname(os.path.abspath(target)) or "."
        os.makedirs(out_dir, exist_ok=True)
        # Copy then edit, then rename: a rebase interrupted halfway must not
        # leave a file with a valid magic and half-rewritten records for
        # something else to pick up. Same shape as merge.py and transcode.py.
        fd, tmp = tempfile.mkstemp(dir=out_dir, prefix=".rebase-", suffix=".vfi")
        os.close(fd)
        try:
            shutil.copyfile(path, tmp)
            umask = os.umask(0)
            os.umask(umask)
            os.chmod(tmp, 0o666 & ~umask)
            detail = _rewrite(tmp, old_root, new_root, new_paths)
            os.replace(tmp, target)
        except BaseException:
            if os.path.exists(tmp):
                os.unlink(tmp)
            raise

    return dict(path=path, out=target, in_place=in_place, videos=n_videos,
                old_root=old_root, new_root=new_root,
                sample=new_paths[0] if new_paths else "", **detail)
