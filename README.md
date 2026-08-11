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

# 2. index it
vlaframe index /clips corpus.vfi
vlaframe info corpus.vfi

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
context and cached file descriptors belong to the process that created them.
Construct one per worker, inside the worker.

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

## Repository layout

```
src/vlaframe/
    transcode.py   function 1 — format conversion
    index.py       function 2 — offline index construction
    reader.py      function 3 — online random access
    _format.py     the .vfi binary format
    cli.py         vlaframe transcode | index | info | get | bench
tests/
    test_correctness.py   the bit-exactness gate
```
