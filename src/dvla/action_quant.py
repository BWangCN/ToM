"""Uniform action quantization (CLAUDE.md §2.2).

Encode:  b = clip(floor((x + 1) / 2 * BINS), 0, BINS-1)
Decode:  x = (b + 0.5) / BINS * 2 - 1   (bin center)

Codebook is shared across action dims. Token order over a chunk is
interleaved per timestep: dx1, dy1, dx2, dy2, ..., dx8, dy8  -> 16 tokens.
"""

import numpy as np

BINS = 128
CHUNK_LEN = 8
ACTION_DIM = 2
TOKENS_PER_CHUNK = CHUNK_LEN * ACTION_DIM  # 16


def encode(x: np.ndarray, bins: int = BINS) -> np.ndarray:
    """Continuous [-1,1] -> bin indices [0, bins-1]. Any shape."""
    x = np.asarray(x, dtype=np.float64)
    b = np.floor((x + 1.0) / 2.0 * bins)
    return np.clip(b, 0, bins - 1).astype(np.int64)


def decode(b: np.ndarray, bins: int = BINS) -> np.ndarray:
    """Bin indices -> bin-center continuous values in (-1, 1)."""
    b = np.asarray(b, dtype=np.float64)
    return (b + 0.5) / bins * 2.0 - 1.0


def chunk_to_bins(chunk: np.ndarray) -> np.ndarray:
    """(H, 2) action chunk -> (16,) interleaved bin sequence."""
    chunk = np.asarray(chunk, dtype=np.float64)
    assert chunk.shape == (CHUNK_LEN, ACTION_DIM), f"bad chunk shape {chunk.shape}"
    return encode(chunk).reshape(-1)  # row-major: dx1, dy1, dx2, dy2, ...


def bins_to_chunk(bins_seq: np.ndarray) -> np.ndarray:
    """(16,) interleaved bin sequence -> (H, 2) bin-center chunk."""
    bins_seq = np.asarray(bins_seq, dtype=np.int64)
    assert bins_seq.shape == (TOKENS_PER_CHUNK,), f"bad bins shape {bins_seq.shape}"
    return decode(bins_seq).reshape(CHUNK_LEN, ACTION_DIM)


def act_token_strs(bins_seq: np.ndarray) -> list[str]:
    return [f"<|act_{int(b)}|>" for b in bins_seq]


def half_bin_width(bins: int = BINS) -> float:
    return 1.0 / bins  # bin width is 2/bins; half-width = 1/bins
