# vlaframe

Random single-frame access over a large H.264 corpus — pick any video out of
millions, pick any frame out of it, get pixels back in about 10 ms without
touching the container at read time.

Built for VLA training dataloaders, where a sample is one randomly chosen frame
of one randomly chosen clip and the working set is far larger than RAM.

```python
from vlaframe import FrameReader

with FrameReader("corpus.vfi") as r:
    img = r.rgb("/data/clips/ep0042_wrist.mp4", 1234)   # HxWx3 uint8
```

The returned pixels are **bit-identical to decord / PyAV**. That is enforced by a
gate, not asserted — see [Correctness](#correctness).

## What it does

| | |
|---|---|
| **1. transcode** | Bring videos into the format the reader is fast on: GOP, CRF, profile, pixel format, optional long-edge cap. Frame-count verified, written atomically. |
| **2. index** | Demux each video once, offline, into a compact memory-mappable `.vfi` file (~8.8 B/frame). After this the container is never parsed again. |
| **3. read** | Decode an arbitrary frame with dependency pruning and a persistent decoder pool. `libavformat` is never invoked. |

## Install

```bash
pip install git+https://github.com/YaoChengTang/VLAFrame.git
```

Everything it needs comes with it — `numpy`, `av`, and a static ffmpeg build via
`imageio-ffmpeg`. **No virtualenv, no system packages, no separate ffmpeg
install.** It drops into an existing conda env or system Python as-is.

Verified end to end (transcode + index + read, bit-exact) on:

| | |
|---|---|
| Python | 3.10, 3.11, 3.13 |
| PyAV | 16.1, 17.0, 17.1, 18.0 |
| numpy | 1.26, 2.2, 2.5 |

`transcode` resolves its ffmpeg from `$FFMPEG_BIN`, then the bundled
`imageio-ffmpeg` build, then `PATH`. The bundled one is preferred over a system
ffmpeg on purpose: distro builds are frequently compiled without libx264, and
that only surfaces part-way through a transcode. Set `$FFMPEG_BIN` to override.

```python
from vlaframe import ffmpeg_bin
print(ffmpeg_bin())      # which binary transcode will actually use
```

Optional extra: `pip install 'vlaframe[image]'` adds Pillow, needed only by
`vlaframe get --out frame.png`.

## Use

### Command line

```bash
# 1. convert a tree (or one file) into the target format
vlaframe transcode /raw /clips --gop 30 --crf 23 --profile high
vlaframe transcode /raw /clips --gop 8 --bframes 0        # faster seeks, more bytes
vlaframe transcode /raw /clips --max-side 640             # cap the longer edge

# 2. index it (shard a large corpus, then merge the shards back)
vlaframe index /clips corpus.vfi
vlaframe merge corpus.vfi 'shards/*.vfi' --verify
vlaframe info corpus.vfi                       # reports whether lookups bisect
vlaframe seal 'shards/*.vfi'                   # upgrade indexes built before v0.1.1
vlaframe rebase corpus.vfi --root /clips       # move stored paths onto a root

# 3. read
vlaframe get corpus.vfi 0 1234 --out frame.png
vlaframe bench corpus.vfi -n 300
```

### Python

```python
from vlaframe import transcode, TranscodeSpec, build_index_from_root, FrameReader

transcode("in.mp4", "out.mp4", TranscodeSpec(gop=30, crf=23, profile="high"))
build_index_from_root("/clips", "corpus.vfi")

r = FrameReader("corpus.vfi", threads=1)       # one per DataLoader worker
img = r.rgb(video_id_or_path, frame_index)     # HxWx3 uint8
```

`FrameReader` is neither thread-safe nor fork-safe once used — the decoder
context belongs to the process that created it. Construct one per worker,
inside the worker.

**It holds no per-video state.** Address a video by path and the lookup bisects
the mmap; nothing is memoised between calls. This is what makes a worker's
memory independent of how much of the corpus it has sampled — see
[Memory](#memory).

## Why it is fast

At 640x480, cold, the path this replaces spends its time roughly half on parsing
the container and half on decoding. Three things remove most of both, and **none
of them change a single byte of video data**:

**The index removes container parsing.** 9.68 ms → 0.07 ms. Afterwards the
lookup, the read and packet construction together are ~0.11 ms, and decoding is
98% of the call.

**Dependency pruning removes most of the decode.** H.264 marks pictures nothing
refers to with `nal_ref_idc == 0`; they can be dropped without disturbing
anything else, and they do not even advance `frame_num`. libavcodec will not do
this — after a seek it decodes the whole prefix — but the index records the flag
offline, so the reader feeds only the references plus the target. Measured 14
packets down to 8, and device reads from 164 KB down to 36 KB.

The flag is read out of the bitstream, never inferred from encoder settings:
x264's default b-pyramid promotes some B-frames to references, so a rule like
"B-frames are disposable" would corrupt exactly those.

**A persistent decoder pool removes decoder setup.** One live decoder per
parameter set — a real corpus has a handful, not thousands. Worth ~0.06 ms on
CPU and ~37 ms on NVDEC.

## Correctness

The failure mode here is silent. A reader that returns a *neighbouring* frame
still returns a plausible image; the only symptom is that pixels stop lining up
with the action or state recorded for that timestep.

**The rule that makes it correct: never count frames, match the output by
timestamp.** With B-frames, decode order is not display order and a decoder
withholds frames by its reorder depth before emitting the first one. Every
frame-indexing bug previously found on this corpus was that delay mistaken for
"the Nth output is the Nth frame" — a fixed +2 offset on 12% of frames from one
decoder, 0–51% alignment from five attempts at another.

Run the gate:

```bash
python tests/test_correctness.py --index corpus.vfi --backend cpu,gpu \
                                 --modes persistent,cold
```

It checks four things, and dropping any one has historically produced a false
pass:

1. **Bit exactness** against libavformat-open-seek-decode.
2. **Keyframe boundaries specifically** — every keyframe k contributes k−1, k,
   k+1. Uniform random indices land mid-GOP and miss reorder bugs; the first
   verification round on this corpus used one file and twelve random indices,
   concluded "identical", and was wrong.
3. **Both decoder lifetimes**, persistent and per-call.
4. **Discriminative power** — on a static shot the neighbours are identical to
   the target, so returning the wrong one still passes. The summary reports how
   many cases could have caught an off-by-one at all.

Current status: **111/111 bit-exact in all four backend × lifetime combinations,
100% discriminative.**

### Two traps this package already walked into

**Synthetic timestamps are not a valid shortcut.** Feeding decode-order integers
instead of real presentation timestamps works with the software decoder, which
reorders to display order and carries each value with its own picture. It does
*not* work with `h264_cuvid`, which stamps its reordered output *positionally*
and pairs every frame with the wrong picture — 63/111 exact, differing by up to
244 per channel, with nothing visibly wrong. The index therefore stores real
timestamps (as a base and a step, so still 0 B/frame).

**NV12 and YUV420P do not reach RGB the same way.** `h264_cuvid` returns NV12 and
the software decoder returns YUV420P. The samples are identical — compare the two
in yuv420p and they are bit-exact — but `to_ndarray("rgb24")` takes a different
swscale chroma-upsampling path for each and the results differ by up to 19 per
channel. The reader normalises to yuv420p first. Large enough to move a model's
inputs; far too small to notice by eye.

## Index format

`.vfi` is a single memory-mapped file. A worker faults in only the pages of the
videos it actually samples.

Measured **8.8 B/frame** — a 2.6-billion-frame corpus indexes to about 23 GB
rather than 76 GB — because three things are derived rather than stored:

| Field | How it is recovered | Saves |
|---|---|---|
| `pts` | base + step, since timestamps are arithmetic in display order (200/200 videos measured) | 8 B/frame |
| `pos` | first offset plus preceding sizes, since packets sit back-to-back in `mdat` (300/300 measured) | 8 B/frame |
| `prev_key` | bisecting the keyframe list, which is ~1/GOP as long | 4 B/frame |

Each is **validated per video at build time**, and any video that violates the
assumption stores an explicit array and sets a flag. Nothing is assumed globally.

### Paths are stored relative to a root

The index records the prefix its videos share once, and each record holds only
the remainder:

```
root    /data/corpus/RoboChallenge/ur5
stored  video_global/00000000-....mp4
```

This matters because `find` matches by exact bytes. An index that stored
absolute paths was a function of where the corpus happened to be mounted and of
what the directories above the videos were called, so renaming a dataset
directory, or mounting the same corpus under a different prefix on another
machine, made **every** lookup miss — with nothing failing at open time, because
the file is still a perfectly valid index.

`find` accepts either form: an absolute path has the root stripped before the
bisection, a relative one is used as is. And the root is overridable at open
time, which is the point — a corpus that moved needs a caller that says so, not
a rebuilt index:

```python
FrameReader("leaf.vfi", root="/new/mount/RoboChallenge/ur5")
```

`off_root == 0` means "no root, the strings are absolute". Those 8 bytes were
reserved-and-zero in the original header, so indexes written before roots
existed keep reading untouched — this is not a format version bump.

To move an existing index onto a root without rebuilding it, see `vlaframe
rebase` below: the paths are ~1.5% of a `.vfi` and nothing in the per-frame
arrays depends on them, so it rewrites that 1.5% and leaves the rest in place.
Measured on a real 148.6 MB / 8,812-video index: **0.36 s**.

```bash
vlaframe rebase leaf.vfi --root /data/corpus/RoboChallenge/ur5   # -> leaf.vfi.rebased
vlaframe rebase 'index_v3_leaves/*.vfi' --root ... --in-place    # cheap and destructive
vlaframe rebase leaf.vfi --to-absolute                           # the inverse
```

## Performance

Measured on this machine — Xeon Silver 4314 @ 2.4 GHz, 32 physical cores — over
640×480 clips at GOP 30, with the pool on tmpfs so storage latency is out of the
picture. Single process, one decode thread:

| | p50 |
|---|---|
| open + seek + decode (what this replaces) | 17.68 ms |
| `FrameReader.frame` — decode only | **9.12 ms** |
| `FrameReader.rgb` — including YUV→RGB | **11.12 ms** |
| `FrameReader.rgb`, 2 decode threads | 10.61 ms |

Two things to be careful about when comparing against other numbers:

- **State the resolution.** The same code takes 30.4 ms at 1280×720 and 10.5 ms
  at 640×480. A mixed pool reports roughly double and looks impossible.
- **State whether RGB conversion is included.** It is ~2 ms and it is real work a
  dataloader has to do somewhere — though if the training pipeline wants GPU
  tensors anyway, doing the conversion there removes it from the CPU budget.

### Threads

More decode threads buy latency at a steep cost in throughput, because CABAC
entropy decoding is inherently serial:

| threads | latency | speedup | core efficiency | per-core throughput |
|---|---|---|---|---|
| 1 | 10.04 ms | 1.00× | 100% | **99.6 /s** |
| 2 | 6.83 ms | 1.47× | 73% | 73.2 /s |
| 4 | **5.66 ms** | 1.77× | 44% | 44.1 /s |
| 8 | 6.57 ms | 1.53× | 19% | 19.0 /s |

Best latency is at 4, not 8 — past that, thread synchronisation costs more than
it saves. For a throughput-bound dataloader, **one thread per worker produces
2.3× more frames per core** than four.

### GPU

`backend="gpu"` uses `h264_cuvid` and is correct, but **is not recommended for
this access pattern**:

- NVDEC is a fixed-function block (7 per B200), not the CUDA cores, and CABAC is
  serial regardless of hardware.
- `flush_buffers()` costs ~10.7 ms on CUVID against microseconds on CPU, and it
  cannot be skipped — after `decode(None)` drains the reorder buffer the context
  refuses further packets until flushed.
- Each `decode()` is a synchronous round trip, ~1.8 ms per packet.
- In the multi-process shape a DataLoader actually uses, it measured **267 ms per
  sample**, 26× worse than CPU: without MPS, processes sharing a GPU time-slice
  their contexts and the switching costs more than the decode.

NVDEC wins on *streams*, where the pipeline stays full. It loses on random single
frames, where it never does.

## Choosing transcode settings

`gop` is the dominant knob and it trades directly against storage — the reader
walks from a keyframe to the target, so cost is linear in that distance. At
640×480: GOP 30 is 6.07 MB/1500 frames and ~10 ms/frame; GOP 8 is 9.21 MB and
~4 ms.

`bframes=0` removes reorder delay entirely at about +5% storage. Worth it if you
are debugging frame alignment or writing your own decoder integration; not
necessary for this package, which handles any reorder depth.

`crf` and `profile` change quality and compatibility, not seek behaviour.

## Memory

A worker's resident memory does not grow with the number of videos it has
sampled. That is a property worth stating explicitly, because the obvious
implementation does not have it: memoising per video looks free on a benchmark
that touches a handful of clips and is fatal on a corpus.

Four memos were removed for this reason. Measured on a 6.98 M video /
2.97 B frame corpus, per DataLoader worker:

| Removed | What it held | Cost |
|---|---|---|
| `FrameReader._rank` | decode-order → display-rank permutation | 8 B/frame = **23.8 GB** |
| `IndexFile._meta` | unpacked `VideoMeta` per video | 474 B/video = **3.3 GB** |
| `IndexFile._by_path` | path → id dict, built in full on first lookup | 107 B/video = **0.75 GB** |
| `FrameReader._fds` | one open mp4 descriptor per video, never evicted | **6.98 M descriptors** |

None was replaced with a bounded cache. The hit rate is not merely low, it is
structurally zero: the sampler is globally shuffled and a run of this size does
not finish an epoch, so a video is touched about once and any per-video cache is
a miss followed by bookkeeping. (The same argument, on the same corpus, is why
the consumer's `decord_pool` backend was deleted.) On a small working set where
the hit rate is *not* zero an LRU does pay — 182 videos, capacity 512, 97.6% hits
— so this is a property of the workload, not a claim that caching is wrong.

What replaces the path → id dict is a bisection of the mmap, enabled by the
`HFLAG_PATHS_SORTED` header flag that the writer now sets after emitting records
in path order. Indexes written before the flag existed are read correctly either
way — an unset flag falls back to the dict — and `vlaframe seal` sets it in
place, after verifying the order, by rewriting 4 header bytes rather than
rebuilding the file.

Measured over 3,000 reads touching 3,200 distinct videos, private anonymous
memory:

```
before   RssAnon  3.4 -> 80.9 MB    +25,816 B/read
after    RssAnon  0.7 ->  0.1 MB       -188 B/read
```

Run the gate:

```bash
python tests/test_memory.py --index corpus.vfi --reads 4000
```

### What this costs, and where

Recomputing is cheaper than the memo it replaces on every axis but one:

| Per read | Cost |
|---|---|
| `_pts` scatter, 900 / 5,000 / 13,000 frames | 3.9 / 14.2 / 107 µs |
| `struct.unpack_from` of one record | 0.4 µs |
| bisection, V = 10,000 (≈14 probes) | 6.3 µs |
| **`os.open` + `os.close`** | **see below** |

The descriptor is the exception, and only on some filesystems. Locally it is
2 µs and invisible. On an NFSv4 export mounted `lookupcache=none` — which is
where this corpus lives — every `open` is a LOOKUP round trip and every `close`
is a CLOSE round trip:

```
open  (fd kept)   2.488 ms
close             1.255 ms
open + close      3.435 ms
```

So on that mount, opening per read costs about **+1 ms per view** against
keeping the descriptor, even with no reuse at all, and more where there is
reuse. Measured end to end at 640x400, p50 per frame: 13.9 ms holding
descriptors, 15.9 ms opening per read. That is the price of bounded memory on
this mount, and it is paid in the mount, not in this package: `lookupcache=pos`
returns `open` to microseconds and makes the trade free. The dentry cache it
enables is reclaimable slab, about 1.2 KB per distinct file touched.


## Repository layout

```
src/vlaframe/
    transcode.py   function 1 — format conversion
    index.py       function 2 — offline index construction
    reader.py      function 3 — online random access
    merge.py       function 4 — combine shards, byte for byte
    rebase.py      function 5 — move stored paths onto a root, without rebuilding
    _format.py     the .vfi binary format
    cli.py         vlaframe transcode | index | merge | rebase | info | seal | get | bench
tests/
    test_correctness.py   the bit-exactness gate
    test_memory.py        the no-per-video-state gate
    test_merge.py         the merge-is-indistinguishable gate
    test_root.py          the path-root gate
```
