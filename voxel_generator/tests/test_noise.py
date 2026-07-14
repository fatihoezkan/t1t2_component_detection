"""Gaussian noise tests."""
import numpy as np

from voxel_simulator.noise import (
    add_gaussian_noise,
    add_noise,
)


def _clean():
    return np.linspace(0.1, 1.0, 64)


def test_shape_and_finite():
    noisy, sigma = add_noise(_clean(), snr=50, rng=np.random.default_rng(0))
    assert noisy.shape == (64,)
    assert np.all(np.isfinite(noisy)) and sigma > 0


def test_reproducible():
    a, _ = add_noise(_clean(), 40, np.random.default_rng(7))
    b, _ = add_noise(_clean(), 40, np.random.default_rng(7))
    np.testing.assert_array_equal(a, b)


def test_direct_sigma_overrides_snr():
    """Passing sigma sets the absolute noise level and ignores SNR."""
    clean = np.linspace(-1.0, 1.0, 4000)
    noisy, sigma = add_gaussian_noise(clean, snr=999, rng=np.random.default_rng(0), sigma=0.1)
    assert sigma == 0.1
    assert abs((noisy - clean).std() - 0.1) < 0.02      # residual std ≈ requested sigma
