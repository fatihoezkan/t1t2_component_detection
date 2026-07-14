#!/usr/bin/env bash
# One-time environment setup on the cluster. Run once on a login node from the repo root:
#   cd ~/t1t2_training && bash slurm/setup_env.sh
#
# Creates a local venv (.venv) and installs a CUDA-enabled torch plus the project deps.
set -euo pipefail

# --- FILL IN: load your cluster's Python/CUDA modules (see the HPC "How To") -----------------
# e.g. module load python/3.11  cuda/12.1
# module load <PYTHON_MODULE>
# module load <CUDA_MODULE>

# --- CUDA wheel index: match your cluster's CUDA (cu121 shown; check `nvidia-smi`) -----------
TORCH_CUDA_INDEX="https://download.pytorch.org/whl/cu121"

python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip

# CUDA torch first (so the generic requirement doesn't pull a CPU wheel), then the rest.
pip install torch --index-url "${TORCH_CUDA_INDEX}"
pip install -r requirements.txt

echo
echo "env ready. sanity check:"
python -c "import torch; print('torch', torch.__version__, '| cuda available:', torch.cuda.is_available())"
echo "activate later with:  source .venv/bin/activate"
