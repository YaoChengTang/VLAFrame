"""vlaframe — random single-frame access over a large H.264 corpus.

Three pieces, in the order they are used:

1. :mod:`vlaframe.transcode` — bring videos into the target format
   (GOP, CRF, profile, optional resolution cap), frame-count verified.
2. :mod:`vlaframe.index` — demux each video once and write a compact,
   memory-mappable ``.vfi`` index (~8 B/frame).
3. :mod:`vlaframe.reader` — decode an arbitrary frame of an arbitrary video
   without ever touching libavformat, using dependency pruning and a
   persistent decoder pool.

    from vlaframe import FrameReader
    with FrameReader("corpus.vfi") as r:
        img = r.rgb(path, 1234)          # HxWx3 uint8, bit-identical to decord
"""
from ._format import (HFLAG_PATHS_SORTED, IndexFile, IndexWriter, VideoMeta,
                      seal_sorted)
from .index import build_index, build_index_from_root, iter_videos, scan_video
from .reader import DecoderPool, FrameReader
from .transcode import (TranscodeResult, TranscodeSpec, count_frames,
                        ffmpeg_bin, probe, target_dims, transcode)

__version__ = "0.1.0"
__all__ = [
    "FrameReader", "DecoderPool",
    "build_index", "build_index_from_root", "iter_videos", "scan_video",
    "transcode", "TranscodeSpec", "TranscodeResult", "probe", "count_frames",
    "target_dims", "ffmpeg_bin",
    "IndexFile", "IndexWriter", "VideoMeta", "seal_sorted",
    "HFLAG_PATHS_SORTED",
    "__version__",
]
