#!/usr/bin/env bash
# Shared setup for every Slurm job: enter the repo, activate the venv, log the environment.
#
# Slurm runs submitted scripts from /var/spool, so jobs source this file by its cluster path.
set -euo pipefail

REPO="/home/fao8402/t1t2_component_detection"
cd "${REPO}"

if [[ ! -f "voxel_generator/data/ti_te_dict.mat" ]]; then
    # The one file the whole pipeline is meaningless without, and it used to be gitignored —
    # so check for it rather than discover it missing halfway through a job.
    echo "FATAL: protocol missing at ${REPO}/voxel_generator/data/ti_te_dict.mat" >&2
    exit 1
fi

# --- environment ------------------------------------------------------------------------
VENV="/home/fao8402/venvs/thesis"
if [[ ! -f "${VENV}/bin/activate" ]]; then
    echo "FATAL: cluster environment not found at ${VENV}" >&2
    exit 1
fi
# shellcheck disable=SC1091
source "${VENV}/bin/activate"
export PYTHONPATH="${REPO}/src:${REPO}/voxel_generator/src:${PYTHONPATH:-}"

# --- log what we actually got -----------------------------------------------------------
# When a run behaves oddly six weeks later, this block is the difference between "we know" and
# "we guess".
echo "=================================================================="
echo "job        : ${SLURM_JOB_NAME:-local} (${SLURM_JOB_ID:-no-slurm})${SLURM_ARRAY_TASK_ID:+ task ${SLURM_ARRAY_TASK_ID}}"
echo "host       : $(hostname)"
echo "repo       : ${REPO}"
echo "venv       : ${VENV}"
echo "started    : $(date -Iseconds)"
echo "git        : $(git -C "${REPO}" rev-parse --short HEAD 2>/dev/null || echo '?')$(git -C "${REPO}" diff --quiet 2>/dev/null || echo ' (DIRTY)')"
echo "protocol   : $(sha256sum voxel_generator/data/ti_te_dict.mat 2>/dev/null | cut -c1-16 || shasum -a 256 voxel_generator/data/ti_te_dict.mat | cut -c1-16)"
python - <<'PY'
import platform, numpy, sys
print(f"python     : {platform.python_version()}  ({sys.executable})")
print(f"numpy      : {numpy.__version__}")
try:
    import torch
    print(f"torch      : {torch.__version__}  cuda_available={torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"gpu        : {torch.cuda.get_device_name(0)}  (cuda {torch.version.cuda})")
except ImportError:
    print("torch      : not installed (fine for CPU-only data generation)")
PY
echo "=================================================================="
