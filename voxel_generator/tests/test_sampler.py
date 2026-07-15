"""Sampler tests — random compartments (T1 > T2, in range), weights, seeding, coverage."""
import numpy as np
import pytest

from voxel_simulator.sampler import (
    MAX_COMP,
    MIN_WEIGHT,
    SPLIT_TRAIN,
    SPLIT_VAL,
    T1_RANGE,
    T2_RANGE,
    sample_random_compartment,
    sample_voxel_spec,
    sample_weights,
    validate_ranges,
    voxel_rng,
)


def test_random_compartment_keeps_t1_above_t2_and_in_range():
    rng = np.random.default_rng(0)
    for _ in range(2000):
        t1, t2 = sample_random_compartment(rng)
        assert t1 > t2                                    # the one physical tie we keep
        assert T1_RANGE[0] <= t1 <= T1_RANGE[1]
        assert T2_RANGE[0] <= t2 <= t1


def test_voxel_spec_shapes_and_t1_gt_t2():
    for n_comp in range(1, MAX_COMP + 1):
        for vid in range(50):
            spec = sample_voxel_spec(vid, n_comp=n_comp, base_seed=0)
            assert spec.n_comp == n_comp
            assert spec.t1.shape == spec.t2.shape == spec.w.shape == (n_comp,)
            assert np.all(spec.t1 > spec.t2)


def test_weights_sum_to_one_and_respect_floor():
    rng = np.random.default_rng(0)
    for n in range(1, MAX_COMP + 1):
        w = sample_weights(n, rng)
        assert abs(w.sum() - 1.0) < 1e-9
        assert w.min() >= MIN_WEIGHT - 1e-12


def test_n_comp_outside_range_raises():
    for bad in (0, MAX_COMP + 1):
        with pytest.raises(ValueError, match="n_comp must be in"):
            sample_voxel_spec(0, n_comp=bad)


# --------------------------------------------------------------------------------------
# Seeding: reproducible, and the collision class the arithmetic seed used to have is gone.
# --------------------------------------------------------------------------------------

def test_same_key_is_bit_reproducible():
    a = sample_voxel_spec(5, n_comp=3, base_seed=0, split_code=SPLIT_TRAIN)
    b = sample_voxel_spec(5, n_comp=3, base_seed=0, split_code=SPLIT_TRAIN)
    np.testing.assert_array_equal(a.t1, b.t1)
    np.testing.assert_array_equal(a.w, b.w)
    assert a.snr == b.snr


def test_splits_and_seeds_and_counts_all_separate_the_stream():
    """Any change to the key must land on a different voxel."""
    base = sample_voxel_spec(5, n_comp=3, base_seed=0, split_code=SPLIT_TRAIN)
    for other in (
        sample_voxel_spec(5, n_comp=3, base_seed=0, split_code=SPLIT_VAL),    # split differs
        sample_voxel_spec(5, n_comp=3, base_seed=1, split_code=SPLIT_TRAIN),  # base seed differs
        sample_voxel_spec(6, n_comp=3, base_seed=0, split_code=SPLIT_TRAIN),  # voxel id differs
    ):
        assert not np.array_equal(base.t1, other.t1)


def test_old_arithmetic_seed_collision_cannot_return():
    """Regression for the collision the arithmetic seed had.

    The old scheme was `master*1_000_003 + voxel_id`, so a split's voxel 10,000,030 landed on the
    exact state of the next split's voxel 0 — i.e. train silently sharing ground truth with val
    once the split grew past the stride. SeedSequence keys cannot alias like that.
    """
    a = sample_voxel_spec(10_000_030, n_comp=2, base_seed=0, split_code=SPLIT_TRAIN)
    b = sample_voxel_spec(0, n_comp=2, base_seed=10, split_code=SPLIT_TRAIN)
    assert not np.array_equal(a.t1, b.t1)


def test_param_stream_ignores_whether_snr_was_drawn():
    """The paired ladder rests on this: pinning the SNR must not disturb the parameter draw.

    SNR lives in its own stream, so a pinned rung and a random-SNR voxel with the same key are the
    same voxel. If SNR were drawn from the parameter stream, every rung would hold different
    compartments and the SNR sweep would confound noise with sampling variation.
    """
    drawn = sample_voxel_spec(7, n_comp=3, base_seed=0)
    pinned = sample_voxel_spec(7, n_comp=3, base_seed=0, snr=20.0)
    np.testing.assert_array_equal(drawn.t1, pinned.t1)
    np.testing.assert_array_equal(drawn.t2, pinned.t2)
    np.testing.assert_array_equal(drawn.w, pinned.w)
    assert pinned.snr == 20.0 and drawn.snr != 20.0


def test_voxel_rng_streams_are_independent():
    from voxel_simulator.sampler import STREAM_NOISE, STREAM_PARAMS

    a = voxel_rng(0, 2, SPLIT_TRAIN, 3, STREAM_PARAMS).standard_normal(8)
    b = voxel_rng(0, 2, SPLIT_TRAIN, 3, STREAM_NOISE).standard_normal(8)
    assert not np.allclose(a, b)


# --------------------------------------------------------------------------------------
# Coverage: the property the old suite could not see.
# --------------------------------------------------------------------------------------

def test_validate_ranges_rejects_infeasible_and_inverted():
    with pytest.raises(ValueError, match="no .T1, T2. with T2 < T1"):
        validate_ranges((50.0, 100.0), (200.0, 3000.0))     # t2_min above t1_max
    for bad in ((0.0, 100.0), (100.0, 50.0)):
        with pytest.raises(ValueError, match="0 < min < max"):
            validate_ranges(bad, T2_RANGE)
        with pytest.raises(ValueError, match="0 < min < max"):
            validate_ranges(T1_RANGE, bad)


def test_infeasible_range_raises_before_numpy_does():
    """A bad range must fail with our message, not numpy's `high - low < 0` from deep inside."""
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError, match="no .T1, T2. with T2 < T1"):
        sample_random_compartment(rng, (50.0, 100.0), (200.0, 3000.0))


def test_log_t2_is_uniform_within_its_feasible_row():
    """Rejection sampling gives a uniform *joint* over the feasible log region.

    This is the test the old suite structurally lacked: every previous check asserted a per-draw
    invariant (T1 > T2, in range), which conditional clipping also satisfied — so the coverage
    distortion sailed through. Here we assert a distribution instead.

    Under the clipped scheme, log-T2 was uniform on [log t2_min, log T1] for *every* T1, so this
    normalized position was U(0,1) in each band but the T1 marginal was flat, oversampling short
    T1. Under rejection, T1's marginal is proportional to its feasible width, and the normalized
    log-T2 position stays U(0,1). We check the latter holds in bands far apart in T1.
    """
    rng = np.random.default_rng(1)
    t1s, t2s = np.empty(20_000), np.empty(20_000)
    for i in range(20_000):
        t1s[i], t2s[i] = sample_random_compartment(rng)

    lo2 = np.log(T2_RANGE[0])
    for lo, hi in ((50.0, 200.0), (1000.0, 4000.0)):
        m = (t1s >= lo) & (t1s < hi)
        assert m.sum() > 500, f"too few samples in T1 band [{lo},{hi})"
        top = np.log(np.minimum(T2_RANGE[1], t1s[m]))
        u = (np.log(t2s[m]) - lo2) / (top - lo2)           # position within the feasible row
        assert 0.0 <= u.min() and u.max() <= 1.0
        for q in (0.25, 0.5, 0.75):                        # uniform => quantile q sits at q
            assert abs(np.quantile(u, q) - q) < 0.05, f"log-T2 not uniform in [{lo},{hi}) at q={q}"


def test_rejection_removes_short_t1_oversampling():
    """The headline coverage fix, stated as a number.

    Clipping made T1 < 200 ms about 31% of draws; a uniform joint over the feasible region puts it
    near 21%. If this ever climbs back, the clipping shortcut has returned.
    """
    rng = np.random.default_rng(2)
    t1s = np.array([sample_random_compartment(rng)[0] for _ in range(20_000)])
    frac_short = float((t1s < 200.0).mean())
    assert 0.18 < frac_short < 0.24, f"T1<200ms fraction {frac_short:.3f} (clipping gave ~0.31)"
