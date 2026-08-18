"""Function 3 — online random single-frame access.

    reader = FrameReader("corpus.vfi")
    img = reader.rgb(path_or_id, frame)          # HxWx3 uint8

One call does: look the frame up in the mmap-ed index, ``pread`` exactly the
keyframe-to-target byte range, feed only the packets that matter to a decoder
that is already alive, and return the frame. libavformat is never invoked.

Measured at 640x480, single process, storage latency removed: 8.00 ms against
17.68 ms for the open-seek-decode path this replaces. Of the 8 ms, 7.7 is
libavcodec; index lookup, the read and packet construction together are 0.3 ms.

The one rule that makes it correct
----------------------------------
**Never count frames. Match the output by timestamp.**

With B-frames the decode order is not the display order, and a decoder withholds
frames by its reorder depth before emitting the first one. Every frame-indexing
bug previously seen on this corpus was that delay being mistaken for "the Nth
output is the Nth frame" -- a fixed +2 offset on 12% of frames from one decoder,
0-51% alignment from five attempts at another. Feeding the range, flushing, and
then selecting the output whose timestamp matches is correct for any reorder
depth, in both a fresh decoder and a reused one, which is what makes the
persistent pool safe.

The timestamps fed in are the video's REAL presentation timestamps, reconstructed
from the base and step the index stores. Synthetic decode-order integers are not
a valid substitute: the software decoder reorders to display order and carries
each value with its own picture, so synthetic values happen to work there, but
``h264_cuvid`` stamps its reordered output POSITIONALLY and pairs every frame
with the wrong picture -- 63/111 exact, differing by up to 244 per channel. The
gate caught it; nothing about the image looks wrong.
"""
from __future__ import annotations

import os
from typing import Iterable

import av
import numpy as np

from ._format import IndexFile, VideoMeta


class DecoderPool:
    """One live decoder per parameter set, per backend.

    Creating a software decoder costs ~0.06 ms and this barely matters; creating
    an NVDEC session costs ~37 ms and it matters enormously. Real corpora have a
    handful of distinct SPS/PPS -- two across the sample used here -- so the pool
    stays tiny either way.

    Between clips the decoder is flushed. That is required, not optional: after
    ``decode(None)`` drains the reorder buffer the context is in an end-of-stream
    state and rejects further packets until it is flushed.
    """

    def __init__(self, backend: str = "cpu", threads: int = 1):
        if backend not in ("cpu", "gpu"):
            raise ValueError(f"backend must be 'cpu' or 'gpu', got {backend!r}")
        self.backend = backend
        self.threads = threads
        self._ctx: dict[int, av.CodecContext] = {}
        self.creations = 0

    def _make(self, extradata: bytes) -> av.CodecContext:
        cc = av.CodecContext.create(
            "h264_cuvid" if self.backend == "gpu" else "h264", "r")
        if extradata:
            cc.extradata = extradata
        if self.backend == "cpu":
            cc.thread_count = self.threads
            cc.thread_type = "AUTO" if self.threads > 1 else "NONE"
        self.creations += 1
        return cc

    def acquire(self, param_set: int, extradata: bytes) -> av.CodecContext:
        cc = self._ctx.get(param_set)
        if cc is None:
            cc = self._make(extradata)
            self._ctx[param_set] = cc
        else:
            cc.flush_buffers()
        return cc

    def close(self) -> None:
        self._ctx.clear()
        self.creations = 0


class FrameReader:
    """Random single-frame access over an indexed corpus.

    Not thread-safe and not fork-safe once used: a decoder context belongs to
    the process that made it. In a DataLoader, construct one per worker.

    Holds no per-video state. An earlier version memoised the decode-order ->
    display-rank permutation and kept every mp4 descriptor it had opened, on the
    reasoning that a worker only pays for the videos it samples. That premise
    does not survive a globally shuffled sampler over a corpus this size: the
    permutation grew to 8 B/frame (23.8 GB on a 2.97 B frame corpus) and the
    descriptors to one per video (6.98 M), in every one of 32 workers.

    Neither was replaced with a bounded cache, for the same reason the
    ``decord_pool`` backend was deleted from the consumer: the hit rate is
    structurally zero, not merely low. A run that cannot finish an epoch touches
    each video about once, so any cache keyed on the video is a miss followed by
    bookkeeping. Measured on a small working set where the hit rate is *not*
    zero (182 videos, an LRU of 512) it does pay -- so this is a property of the
    workload, not a claim that caching is wrong.

    What that leaves is one ``os.open`` per read. On a local filesystem it is
    ~2 us and invisible. On the NFS export this corpus lives on it is 3.1 ms,
    because that mount sets ``lookupcache=none`` and every open costs a LOOKUP
    round trip -- a tax the previous version paid too, on every first touch, and
    one no cache can avoid at ~1 touch per video. It is fixed at the mount, not
    here.
    """

    def __init__(self, index_path: str, *, backend: str = "cpu", threads: int = 1,
                 persistent: bool = True):
        self.index = IndexFile(index_path)
        self.backend = backend
        self.persistent = persistent
        self.pool = DecoderPool(backend, threads)

    # ── addressing ────────────────────────────────────────────────────
    def __len__(self) -> int:
        return self.index.n_videos

    def vid_of(self, path: str) -> int:
        vid = self.index.find(path)
        if vid is None:
            raise KeyError(f"{path} is not in this index")
        return vid

    def _resolve(self, video: int | str) -> VideoMeta:
        return self.index.meta(video if isinstance(video, int) else self.vid_of(video))

    def n_frames(self, video: int | str) -> int:
        return self._resolve(video).n_frames

    def resolution(self, video: int | str) -> tuple[int, int]:
        m = self._resolve(video)
        return m.width, m.height

    def paths(self) -> list[str]:
        return [self.index.meta(i).path for i in range(self.index.n_videos)]

    # ── plumbing ──────────────────────────────────────────────────────
    def _pts(self, m: VideoMeta) -> np.ndarray:
        """Presentation timestamp per packet, in decode order.

        The scatter below inverts the display permutation; it costs 3.9 us on a
        900-frame clip and 14.2 us on a 5000-frame one, which is why it is not
        worth memoising (see the class docstring).
        """
        explicit = self.index.timestamps(m)
        if explicit is not None:
            return explicit
        disp = self.index.by_display(m).astype(np.int64)
        rank = np.empty(m.n_frames, dtype=np.int64)
        rank[disp] = np.arange(m.n_frames, dtype=np.int64)
        return m.pts_base + rank * m.pts_step

    def _plan(self, m: VideoMeta, frame: int) -> tuple[int, int, list[tuple[int, int, int]]]:
        """Work out what to read and what to feed.

        Returns ``(byte_lo, byte_hi, feed)`` where `feed` is
        ``[(offset_within_buffer, size, packet_index), ...]`` in decode order:
        every reference packet from the governing keyframe up to the target,
        plus the target itself.
        """
        if not 0 <= frame < m.n_frames:
            raise IndexError(f"frame {frame} out of range for {m.path} "
                             f"({m.n_frames} frames)")
        sizes = self.index.sizes(m)
        disp = self.index.by_display(m)
        keys = self.index.keys(m)
        refbits = self.index.refbits(m)
        explicit = self.index.positions(m)

        tgt = int(disp[frame])
        # The governing keyframe is the last one at or before the target in
        # decode order; the keyframe list is ~1/GOP the length of a per-frame
        # array, so bisecting it is cheaper than storing prev_key per frame.
        k = int(keys[np.searchsorted(keys, tgt, side="right") - 1])

        # int64 throughout: sizes are uint32 and a naive sum wraps on long clips.
        if explicit is not None:
            starts = explicit[k:tgt + 1].astype(np.int64)
        else:
            cum = np.cumsum(sizes[:tgt + 1].astype(np.int64))
            starts = m.pos0 + np.concatenate([[0], cum[:-1]])[k:tgt + 1]
        lo = int(starts[0])
        hi = int(starts[-1]) + int(sizes[tgt])

        feed: list[tuple[int, int, int]] = []
        for j, i in enumerate(range(k, tgt + 1)):
            # Skip pictures nothing refers to; they cannot affect the target.
            if i == tgt or (refbits[i >> 3] >> (i & 7)) & 1:
                feed.append((int(starts[j]) - lo, int(sizes[i]), i))
        return lo, hi, feed

    @staticmethod
    def _normalize(frame):
        """Put every backend's output in one pixel format before conversion.

        h264_cuvid returns NV12 (interleaved chroma) and the software decoder
        returns YUV420P (planar). The samples are identical -- comparing the two
        in yuv420p is bit-exact -- but the two reach RGB through different
        swscale chroma-upsampling paths and the results differ by up to 19 per
        channel. Large enough to move a model's inputs, small enough that nobody
        would ever catch it by eye.
        """
        return frame if frame.format.name == "yuv420p" \
            else frame.reformat(format="yuv420p")

    # ── the call ──────────────────────────────────────────────────────
    def frame(self, video: int | str, frame: int):
        """Decode one frame and return it as an ``av.VideoFrame`` (yuv420p)."""
        m = self._resolve(video)
        lo, hi, feed = self._plan(m, frame)
        # try/finally, not a bare close after the read: a read that raises would
        # otherwise leak the descriptor, and at corpus scale that is an EMFILE a
        # few thousand samples later rather than an obvious failure here.
        fd = os.open(m.path, os.O_RDONLY)
        try:
            buf = os.pread(fd, hi - lo, lo)
        finally:
            os.close(fd)

        extradata = self.index.extradata(m.param_set)
        cc = self.pool.acquire(m.param_set, extradata) if self.persistent \
            else self.pool._make(extradata)

        pts = self._pts(m)
        target_ts = int(pts[feed[-1][2]])
        for off, sz, i in feed:
            pkt = av.Packet(buf[off:off + sz])
            pkt.pts = pkt.dts = int(pts[i])
            for fr in cc.decode(pkt):
                if fr.pts == target_ts:
                    return self._normalize(fr)
        # Still inside the reorder buffer; draining is what brings it out. Its
        # absence is precisely the classic fixed-offset bug.
        for fr in cc.decode(None):
            if fr.pts == target_ts:
                return self._normalize(fr)
        raise KeyError(f"{m.path}: frame {frame} was never emitted "
                       f"(fed {len(feed)} packets)")

    def rgb(self, video: int | str, frame: int) -> np.ndarray:
        """The call a dataloader wants: HxWx3 uint8, bit-identical to decord."""
        return self.frame(video, frame).to_ndarray(format="rgb24")

    def rgb_batch(self, items: Iterable[tuple[int | str, int]]) -> list[np.ndarray]:
        """Convenience over independent (video, frame) pairs.

        Sequential on purpose. Interleaving them across decoder contexts inside
        one process was measured slower, not faster, because every decode call is
        synchronous -- the parallelism a DataLoader gets comes from its worker
        processes, not from inside one.
        """
        return [self.rgb(v, f) for v, f in items]

    # ── lifecycle ─────────────────────────────────────────────────────
    def close(self) -> None:
        self.pool.close()
        self.index.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
