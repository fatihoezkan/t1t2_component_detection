"""Tests: physics forward model and protocol loading."""

import numpy as np
import pytest

from voxel_simulator.physics import simulate_clean_signal
from voxel_simulator.protocol import Protocol, load_protocol


def test_protocol_load():
    proto = load_protocol()
    assert proto.n_points == 64
    assert len(np.unique(proto.ti)) == 8
    assert len(np.unique(proto.te)) == 8
    assert proto.tr == 20000.0
    assert proto.ti.min() > 0
    assert proto.te.min() > 0


def test_single_compartment_manual():
    """Forward model must match hand-computed values for one compartment."""
    proto = Protocol(
        ti=np.array([500.0, 1000.0, 2000.0]),
        te=np.array([10.0, 50.0, 100.0]),
        tr=20000.0,
    )
    T1, T2 = 1000.0, 80.0
    expected = (1 - 2*np.exp(-proto.ti/T1) + np.exp(-proto.tr/T1)) * np.exp(-proto.te/T2)

    got = simulate_clean_signal(proto, t1=np.array([T1]), t2=np.array([T2]), w=np.array([1.0]))
    np.testing.assert_allclose(got, expected, rtol=1e-12)


def test_linearity_multi_compartment():
    """Multi-compartment signal must equal the weighted sum of single signals."""
    proto = load_protocol()
    t1 = np.array([830.0, 4000.0])
    t2 = np.array([80.0, 2000.0])
    w  = np.array([0.7, 0.3])

    combined = simulate_clean_signal(proto, t1, t2, w)
    s1 = simulate_clean_signal(proto, t1[:1], t2[:1], np.array([1.0]))
    s2 = simulate_clean_signal(proto, t1[1:], t2[1:], np.array([1.0]))
    np.testing.assert_allclose(combined, 0.7 * s1 + 0.3 * s2, rtol=1e-12)


def test_signal_finite_and_shape():
    proto = load_protocol()
    s = simulate_clean_signal(proto, np.array([1000.0]), np.array([80.0]), np.array([1.0]))
    assert s.shape == (64,)
    assert np.all(np.isfinite(s))


def test_invalid_inputs_rejected():
    proto = load_protocol()
    with pytest.raises(ValueError):
        simulate_clean_signal(proto, np.array([-1.0]), np.array([80.0]), np.array([1.0]))
    with pytest.raises(ValueError):
        simulate_clean_signal(proto, np.array([1000.0, 500.0]), np.array([80.0, 20.0]), np.array([0.5, 0.3]))
    with pytest.raises(ValueError):
        simulate_clean_signal(proto, np.array([1000.0]), np.array([80.0]), np.array([-0.5]))


def test_zero_crossing_inversion_recovery():
    """Single-compartment IR zero crossing should be near TI ~ T1 * ln 2 (when TR >> T1)."""
    proto = Protocol(
        ti=np.array([1000.0 * np.log(2)]),
        te=np.array([0.001]),   # essentially no T2 decay
        tr=20000.0,
    )
    s = simulate_clean_signal(proto, np.array([1000.0]), np.array([80.0]), np.array([1.0]))
    # at TI = T1 ln2 and TR >> T1, S ≈ 0 (within ~7e-3 due to e^{-TR/T1} residual)
    assert abs(s[0]) < 0.01
