#!/usr/bin/env bash
#SBATCH --job-name=t1t2_train
#SBATCH --partition=advance
#SBATCH --qos=advance
#SBATCH --account=advance
#SBATCH --gres=gpu:nvidia_a100_80gb_pcie_3g.39gb:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=slurm/logs/train_%x_%j.out
#SBATCH --error=slurm/logs/train_%x_%j.out
#
# Train + evaluate one experiment on a GPU node.
#
#     sbatch slurm/train.slurm                          # configs/cluster.yaml
#
# PREREQUISITES — both are gates, not suggestions:
#   1. slurm/smoke.slurm passed (CUDA works, resume works, the epoch budget fits the wall)
#   2. slurm/audit.slurm passed AND a human read results/data_audit/<name>/
#
# If the 24 h wall is hit, just resubmit: the run resumes from last.pt with its optimizer, epoch,
# history and early-stopping state intact. Resume refuses to continue if the config or dataset
# changed, so a resubmit cannot silently blend two experiments.
#
# 8 CPUs is for dataloading, not for Hungarian matching, which was measured not to be the
# bottleneck. The first run is only configs/cluster.yaml; scaling arms come after it is stable.

source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

CFG="${1:-configs/cluster.yaml}"

if [[ ! -f "${CFG}" ]]; then
    echo "FATAL: no config at ${CFG}" >&2
    exit 1
fi

# The audit is the gate. Refuse to burn a GPU allocation on unaudited data.
DATA_NAME="$(python - "${CFG}" <<'PY'
import sys
sys.path.insert(0, "src")
from pathlib import Path
from t1t2.config import load_config
p = load_config(sys.argv[1]).data.train_path
p = [p] if isinstance(p, str) else p
print(Path(p[0]).parent.parent.name)      # data/baseline_100k/n1/train.parquet -> baseline_100k
PY
)"
AUDIT="${REPO}/results/data_audit/${DATA_NAME}/audit_summary.json"
if [[ ! -f "${AUDIT}" ]]; then
    echo "FATAL: ${DATA_NAME} has not been audited (${AUDIT} missing)." >&2
    echo "       Run: sbatch slurm/audit.slurm" >&2
    exit 1
fi
if ! python -c "import json,sys; sys.exit(0 if json.load(open('${AUDIT}'))['passed'] else 1)"; then
    echo "FATAL: the audit for ${DATA_NAME} did not pass. Do not train on this data." >&2
    exit 1
fi
echo "audit gate: ${DATA_NAME} passed"

echo "training ${CFG}"
srun python -m t1t2.experiment --config "${CFG}"

echo
echo "done at $(date -Iseconds)"
