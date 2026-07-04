"""Quantizer round-trip (CLAUDE.md §2.2 / §7.3)."""

import numpy as np

from dvla.action_quant import (BINS, bins_to_chunk, chunk_to_bins, decode,
                               encode, half_bin_width)


def test_roundtrip_max_error():
    rng = np.random.default_rng(0)
    x = rng.uniform(-1, 1, size=(10000,))
    err = np.abs(decode(encode(x)) - x).max()
    assert err <= half_bin_width() + 1e-12, err


def test_edges():
    assert encode(np.array([-1.0])) == 0
    assert encode(np.array([1.0])) == BINS - 1  # clipped into the top bin
    assert encode(np.array([0.0])) == BINS // 2
    x = decode(np.arange(BINS))
    assert (x > -1).all() and (x < 1).all()
    assert (np.diff(x) > 0).all()  # monotone bin centers


def test_chunk_interleave_inverse():
    rng = np.random.default_rng(1)
    chunk = rng.uniform(-1, 1, size=(8, 2))
    bins = chunk_to_bins(chunk)
    assert bins.shape == (16,)
    # interleaved per timestep: dx1, dy1, dx2, dy2, ...
    assert bins[0] == encode(chunk[0, 0]) and bins[1] == encode(chunk[0, 1])
    assert bins[2] == encode(chunk[1, 0])
    back = bins_to_chunk(bins)
    assert np.abs(back - chunk).max() <= half_bin_width() + 1e-12
