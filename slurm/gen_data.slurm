#!/usr/bin/env bash
#SBATCH --job-name=t1t2_gen
#SBATCH --partition=debugging
#SBATCH --qos=debugging
#SBATCH --account=debugging
#SBATCH --array=1-4                  # one task per compartment count -> data/full_1to4/n<k>/
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=slurm/logs/gen_%A_%a.out
#SBATCH --error=slurm/logs/gen_%A_%a.out
#
# Generate the balanced 1M-voxel dataset: 250k train per compartment count, 25k val, 25k test,
# 12.5k per fixed-SNR rung. One array task per count, so the four run in parallel and a failure
# costs only one of them.
#
#     sbatch slurm/gen_data.slurm
#
# Writes to the shared home, never /fast_storage — that is node-local, so the training job would
# land on a different node and find nothing.
#
# Then: sbatch slurm/audit.slurm   (no training until the audit is green and read)

source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

N_COMP="${SLURM_ARRAY_TASK_ID}"
OUT="${REPO}/data/full_1to4/n${N_COMP}"

echo "generating n_comp=${N_COMP} -> ${OUT}"
cd "${REPO}/voxel_generator"

# No --overwrite: if the data is already there, stop rather than silently spend two hours
# replacing a dataset something downstream may already have been audited against.
srun python run_generator.py \
    --n-comp "${N_COMP}" \
    --seed 0 \
    --out-dir "${OUT}" \
    --n-train 250000 \
    --n-val 25000 \
    --n-test 25000 \
    --n-per-snr 12500 \
    --snr-ladder 20 40 60 100 150

echo
echo "finished n_comp=${N_COMP} at $(date -Iseconds)"
ls -la "${OUT}"
