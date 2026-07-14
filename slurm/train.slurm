#!/usr/bin/env bash
#SBATCH --job-name=t1t2-train
#SBATCH --partition=<GPU_PARTITION>       # <-- FILL IN (a GPU partition)
### #SBATCH --account=<ACCOUNT>           # <-- uncomment + fill if required
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8                  # Hungarian matching runs on CPU — give it cores
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=slurm/logs/%x_%j.out
#SBATCH --error=slurm/logs/%x_%j.err
#
# Train + evaluate one experiment on a GPU node. Resumes from the last checkpoint if present,
# so if the job hits the --time limit you can just re-submit to continue.
set -euo pipefail
mkdir -p slurm/logs

# module load <PYTHON_MODULE>            # <-- match setup_env.sh
# module load <CUDA_MODULE>
source .venv/bin/activate

CONFIG=${1:-configs/cluster.yaml}         # override: sbatch slurm/train.slurm configs/other.yaml
echo "training $CONFIG"
PYTHONPATH=src python -m t1t2.experiment --config "$CONFIG"

echo "done."
