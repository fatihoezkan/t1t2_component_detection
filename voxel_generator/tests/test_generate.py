"""Generation tests: readable voxel pipeline, stable row schema, fixed-n families, safe writes."""

import json

import numpy as np
import pandas as pd
import pytest

from voxel_simulator.generate import (
    DatasetFamilyConfig,
    build_dataset_jobs,
    generate_dataset,
    generate_dataset_family,
    generate_one,
    generate_voxel,
    voxel_to_row,
)
from voxel_simulator.protocol import load_protocol
from voxel_simulator.sampler import MAX_COMP, SPLIT_SNR_LADDER


def test_generate_voxel_exposes_model_input_signal():
    proto = load_protocol()
    voxel = generate_voxel(0, n_comp=2, protocol=proto, noise_sigma=0.1)

    assert voxel.signal.shape == (proto.n_points,)
    assert voxel.sigma == 0.1
    assert np.all(np.isfinite(voxel.signal))


def test_generate_one_matches_voxel_to_row_schema():
    proto = load_protocol()
    voxel = generate_voxel(1, n_comp=2, protocol=proto, noise_sigma=0.1)

    row_from_parts = voxel_to_row(voxel, proto, noise_sigma=0.1)
    row_direct = generate_one(1, n_comp=2, protocol=proto, noise_sigma=0.1)

    assert row_from_parts.keys() == row_direct.keys()
    for key in row_direct:
        if isinstance(row_direct[key], float) and np.isnan(row_direct[key]):
            assert np.isnan(row_from_parts[key])
        else:
            assert row_from_parts[key] == row_direct[key]


def test_n_comp_reaches_every_row():
    """Regression for the bug that made --n-comp a no-op.

    generate_voxel used to be called positionally, so the compartment count never reached the
    sampler: the flag looked like it worked while every file quietly kept the default mix. The
    whole per-n dataset design rests on this, and nothing else would have caught it until the
    audit — after a million voxels.
    """
    proto = load_protocol()
    for n_comp in range(1, MAX_COMP + 1):
        df = generate_dataset(20, n_comp=n_comp, protocol=proto)
        assert (df.n_comp == n_comp).all(), f"asked for n_comp={n_comp}, got {set(df.n_comp)}"


def test_schema_width_is_fixed_regardless_of_n_comp():
    """Per-n files must share one schema or they cannot be concatenated into one dataset."""
    proto = load_protocol()
    frames = {n: generate_dataset(5, n_comp=n, protocol=proto) for n in range(1, MAX_COMP + 1)}
    widths = {n: df.shape[1] for n, df in frames.items()}
    assert len(set(widths.values())) == 1, f"schema width varies with n_comp: {widths}"
    assert next(iter(widths.values())) == 4 + 3 * MAX_COMP + proto.n_points

    for n, df in frames.items():
        for i in range(1, MAX_COMP + 1):
            filled = df[[f"T1_{i}", f"T2_{i}", f"w_{i}"]].notna().all(axis=1)
            assert filled.all() if i <= n else (~filled).all(), f"padding wrong at n={n}, slot {i}"


def test_weights_and_t1_gt_t2_in_rows():
    df = generate_dataset(200, n_comp=3, protocol=load_protocol())
    for i in range(1, 4):
        assert (df[f"T1_{i}"] > df[f"T2_{i}"]).all()
    w = df[[f"w_{i}" for i in range(1, MAX_COMP + 1)]].sum(axis=1)
    np.testing.assert_allclose(w, 1.0, atol=1e-9)


# --------------------------------------------------------------------------------------
# Paired fixed-SNR ladder
# --------------------------------------------------------------------------------------

def test_ladder_jobs_share_split_and_pin_snr():
    jobs = {j.name: j for j in build_dataset_jobs(DatasetFamilyConfig(n_comp=2))}
    rungs = [j for name, j in jobs.items() if name.startswith("test_snr")]
    assert rungs, "no ladder jobs built"
    assert all(j.split_code == SPLIT_SNR_LADDER for j in rungs)
    assert all(j.snr is not None for j in rungs)
    assert jobs["train"].snr is None                       # train draws its SNR per voxel


def test_ladder_rungs_share_ground_truth_and_z(tmp_path):
    """The ladder must be a paired comparison: same voxels, same noise pattern, only amplitude.

    Otherwise every rung holds different compartments and the performance-vs-SNR curve confounds
    the SNR effect with sampling variation across rungs.
    """
    cfg = DatasetFamilyConfig(out_dir=tmp_path, n_comp=2, n_train=0, n_val=0, n_test=0,
                              n_per_snr=40, snr_ladder=(20, 150))
    generate_dataset_family(cfg, verbose=False)

    a = pd.read_parquet(tmp_path / "test_snr20.parquet")
    b = pd.read_parquet(tmp_path / "test_snr150.parquet")

    gt = ["n_comp"] + [f"{p}_{i}" for p in ("T1", "T2", "w") for i in range(1, MAX_COMP + 1)]
    pd.testing.assert_frame_equal(a[gt], b[gt])            # ground truth is float64: exact

    # Recovered z: signals are stored float32, so 1/sigma amplifies storage rounding (~2e-5 at
    # SNR 150). Compare with a tolerance rather than for equality.
    from voxel_simulator.physics import simulate_clean_signal
    proto = load_protocol()
    cols = [f"S_{i+1}" for i in range(proto.n_points)]
    for r in range(len(a)):
        n = int(a.n_comp.iloc[r])
        t1 = a[[f"T1_{i+1}" for i in range(n)]].iloc[r].to_numpy(float)
        t2 = a[[f"T2_{i+1}" for i in range(n)]].iloc[r].to_numpy(float)
        w = a[[f"w_{i+1}" for i in range(n)]].iloc[r].to_numpy(float)
        clean = simulate_clean_signal(proto, t1, t2, w)
        za = (a[cols].iloc[r].to_numpy(float) - clean) / a.sigma.iloc[r]
        zb = (b[cols].iloc[r].to_numpy(float) - clean) / b.sigma.iloc[r]
        np.testing.assert_allclose(za, zb, rtol=0, atol=1e-4)

    assert (a.sigma > b.sigma).all()                       # lower SNR => more noise


# --------------------------------------------------------------------------------------
# Config validation and safe writes
# --------------------------------------------------------------------------------------

def test_config_rejects_bad_n_comp_and_ranges():
    with pytest.raises(ValueError, match="n_comp must be in"):
        DatasetFamilyConfig(n_comp=MAX_COMP + 1)
    with pytest.raises(ValueError, match="no .T1, T2. with T2 < T1"):
        DatasetFamilyConfig(n_comp=1, t1_range=(50.0, 100.0), t2_range=(200.0, 3000.0))


def test_config_rejects_colliding_ladder_names():
    """int(snr) names them, so 20.2 and 20.8 would both be test_snr20 and one would vanish."""
    with pytest.raises(ValueError, match="duplicate output names"):
        DatasetFamilyConfig(n_comp=1, snr_ladder=(20.2, 20.8))


def test_existing_output_is_not_clobbered(tmp_path):
    cfg = DatasetFamilyConfig(out_dir=tmp_path, n_comp=1, n_train=5, n_val=0, n_test=0,
                              n_per_snr=0, snr_ladder=())
    generate_dataset_family(cfg, verbose=False)
    before = (tmp_path / "train.parquet").read_bytes()

    with pytest.raises(FileExistsError, match="already exist"):
        generate_dataset_family(cfg, verbose=False)
    assert (tmp_path / "train.parquet").read_bytes() == before      # untouched

    generate_dataset_family(cfg.__class__(**{**cfg.__dict__, "overwrite": True}), verbose=False)
    assert (tmp_path / "train.parquet").exists()


def test_no_tmp_files_left_behind(tmp_path):
    cfg = DatasetFamilyConfig(out_dir=tmp_path, n_comp=1, n_train=5, n_val=0, n_test=0,
                              n_per_snr=0, snr_ladder=())
    generate_dataset_family(cfg, verbose=False)
    assert not list(tmp_path.glob("*.tmp")), "atomic write left a temp file behind"


def test_manifest_records_provenance(tmp_path):
    cfg = DatasetFamilyConfig(out_dir=tmp_path, n_comp=2, n_train=5, n_val=0, n_test=0,
                              n_per_snr=0, snr_ladder=())
    generate_dataset_family(cfg, verbose=False)
    m = json.loads((tmp_path / "manifest.json").read_text())

    assert m["n_comp"] == 2 and m["max_comp"] == MAX_COMP
    assert m["splits"]["train"]["rows"] == 5
    assert len(m["protocol_sha256"]) == 64
    assert set(m["dependencies"]) == {"python", "numpy", "pandas", "pyarrow"}
    assert "commit" in m["git"]
