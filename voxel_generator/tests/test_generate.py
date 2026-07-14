"""Generation tests: readable voxel pipeline and stable row schema."""

import numpy as np

from voxel_simulator.generate import generate_one, generate_voxel, voxel_to_row
from voxel_simulator.protocol import load_protocol


def test_generate_voxel_exposes_model_input_signal():
    proto = load_protocol()
    voxel = generate_voxel(0, master_seed=0, protocol=proto, noise_sigma=0.1)

    assert voxel.signal.shape == (proto.n_points,)
    assert voxel.sigma == 0.1
    assert np.all(np.isfinite(voxel.signal))


def test_generate_one_matches_voxel_to_row_schema():
    proto = load_protocol()
    voxel = generate_voxel(1, master_seed=0, protocol=proto, noise_sigma=0.1)

    row_from_parts = voxel_to_row(voxel, proto, noise_sigma=0.1)
    row_direct = generate_one(1, master_seed=0, protocol=proto, noise_sigma=0.1)

    assert row_from_parts.keys() == row_direct.keys()
    for key in row_direct:
        if isinstance(row_direct[key], float) and np.isnan(row_direct[key]):
            assert np.isnan(row_from_parts[key])
        else:
            assert row_from_parts[key] == row_direct[key]
