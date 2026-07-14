"""Sampler tests — random compartments (T1 > T2, in range) and basic weight sanity."""
import numpy as np

from voxel_simulator.sampler import (
    MIN_WEIGHT,
    T1_RANGE,
    T2_RANGE,
    sample_random_compartment,
    sample_voxel_spec,
    sample_weights,
)


def test_random_compartment_keeps_t1_above_t2_and_in_range():
    rng = np.random.default_rng(0)
    for _ in range(2000):
        t1, t2 = sample_random_compartment(rng)
        assert t1 > t2                                    # the one physical tie we keep
        assert T1_RANGE[0] <= t1 <= T1_RANGE[1]
        assert T2_RANGE[0] <= t2 <= t1


def test_voxel_spec_shapes_and_t1_gt_t2():
    for vid in range(300):
        spec = sample_voxel_spec(vid, master_seed=0)
        assert 1 <= spec.n_comp <= 3
        assert spec.t1.shape == spec.t2.shape == spec.w.shape == (spec.n_comp,)
        assert np.all(spec.t1 > spec.t2)


def test_weights_sum_to_one_and_respect_floor():
    rng = np.random.default_rng(0)
    for n in (1, 2, 3):
        w = sample_weights(n, rng)
        assert abs(w.sum() - 1.0) < 1e-9
        assert w.min() >= MIN_WEIGHT - 1e-12


def test_different_master_seeds_give_different_voxels():
    """Splits stay leakage-free because a different master seed draws different voxels."""
    a = sample_voxel_spec(5, master_seed=0)
    b = sample_voxel_spec(5, master_seed=10)
    assert not np.array_equal(a.t1, b.t1)
