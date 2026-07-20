#!/usr/bin/env bash
#SBATCH --job-name=t1t2_smoke
#SBATCH --partition=debugging
#SBATCH --qos=debugging
#SBATCH --account=debugging
#SBATCH --gres=gpu:1g.10gb:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=slurm/logs/smoke_%j.out
#SBATCH --error=slurm/logs/smoke_%j.out
#
# Two epochs on CUDA, then a checkpoint/resume test, then a wall-time projection.
#
#     sbatch slurm/smoke.slurm
#
# The projection is the point. Measure the configured baseline on the real GPU and data before
# spending the real allocation; it honors train_limit_per_path rather than projecting against all
# rows that happen to exist in the parquet files.

source "/home/fao8402/t1t2_component_detection/slurm/_common.sh"

CFG="${1:-configs/cluster.yaml}"
RUN="${REPO}/results/_smoke"
rm -rf "${RUN}"

echo "### 1/3 two epochs on CUDA (${CFG})"
srun python -m t1t2.experiment --config "${CFG}" \
    --max-epochs 2 --limit 8192 --results-dir "${RUN}" --no-resume

echo
echo "### 2/3 resume: continue to epoch 4 from last.pt"
srun python -m t1t2.experiment --config "${CFG}" \
    --max-epochs 4 --limit 8192 --results-dir "${RUN}"

echo
echo "### 3/3 throughput projection"
python - "${RUN}" "${CFG}" <<'PY'
import json, sys
from pathlib import Path

run, cfg_path = Path(sys.argv[1]), sys.argv[2]
hist = json.loads((run / "history.json").read_text())

assert len(hist) == 4, f"resume broken: history has {len(hist)} epochs, expected 4"
assert [h["epoch"] for h in hist] == [0, 1, 2, 3], "resume renumbered the epochs"
assert (run / "checkpoints" / "best.pt").exists(), "best.pt was not written"
print("resume + best.pt OK")

sys.path.insert(0, "src")
from t1t2.config import load_config
cfg = load_config(cfg_path)

limit, epochs = 8192, cfg.train.epochs
# Steady state: skip epoch 0, which carries warm-up and cudnn autotuning.
per_epoch = sum(h["seconds"] for h in hist[1:]) / len(hist[1:])
per_voxel = per_epoch / limit

import pandas as pd
per_path = cfg.data.train_limit_per_path
train_voxels = sum(
    min(len(pd.read_parquet(p, columns=["n_comp"])), per_path)
    if per_path is not None else len(pd.read_parquet(p, columns=["n_comp"]))
    for p in cfg.data.train_path
)
proj_epoch = per_voxel * train_voxels
proj_total = proj_epoch * epochs / 3600

print(f"\nmeasured : {per_epoch:.1f} s/epoch at {limit:,} voxels -> {per_voxel*1e6:.1f} us/voxel")
print(f"baseline : {train_voxels:,} train voxels")
print(f"projected: {proj_epoch:.0f} s/epoch, {proj_total:.1f} h for the {epochs}-epoch cap")
print(f"budget   : 24 h -> {'FITS' if proj_total < 24 else 'DOES NOT FIT'}")
if proj_total >= 24:
    print("\n  The 24 h wall will be hit. That is survivable (the job resumes from last.pt on")
    print("  resubmit) and early stopping will likely end it sooner anyway — but plan for it")
    print("  rather than discover it at hour 24.")
PY

echo
echo "smoke passed. Next: sbatch slurm/train.slurm"
