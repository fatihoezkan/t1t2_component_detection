#!/usr/bin/env bash
#SBATCH --job-name=t1t2-gendata
#SBATCH --partition=<CPU_PARTITION>       # <-- FILL IN (a CPU partition; no GPU needed)
### #SBATCH --account=<ACCOUNT>           # <-- uncomment + fill if your cluster requires it
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=slurm/logs/%x_%j.out
#SBATCH --error=slurm/logs/%x_%j.err
#
# Generate the full synthetic dataset ON the cluster (so the big parquet files never need
# uploading). Writes data/full/{train,val,test}.parquet + a fixed-SNR test ladder.
set -euo pipefail
mkdir -p slurm/logs

# module load <PYTHON_MODULE>            # <-- match setup_env.sh
source .venv/bin/activate

cd voxel_generator
PYTHONPATH=src python run_generator.py \
    --out-dir ../data/full \
    --n-train 1000000 --n-val 100000 --n-test 100000 \
    --n-per-snr 50000 \
    --snr-min 20 --snr-max 150 \
    --snr-ladder 20 40 60 100 150

echo "done -> data/full/"
