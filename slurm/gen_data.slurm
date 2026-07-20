#!/usr/bin/env bash
#SBATCH --job-name=t1t2_gen
#SBATCH --partition=debugging
#SBATCH --qos=debugging
#SBATCH --account=debugging
#SBATCH --array=1-3                  # one task per baseline compartment count
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=slurm/logs/gen_%A_%a.out
#SBATCH --error=slurm/logs/gen_%A_%a.out
#
# Generate the baseline-sized family: 33,333 train, 3,333 val, 3,333 test and 1,667 per fixed-SNR
# rung for each of n=1..3. That is 99,999 training voxels and 145,002 artifacts in total.
#
#     sbatch slurm/gen_data.slurm
#
# Writes to the shared home, never /fast_storage — that is node-local, so the training job would
# land on a different node and find nothing.
#
# Then: sbatch slurm/audit.slurm   (no training until the audit is green and read)

source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

N_COMP="${SLURM_ARRAY_TASK_ID}"
OUT="${REPO}/data/baseline_100k/n${N_COMP}"

echo "generating n_comp=${N_COMP} -> ${OUT}"
cd "${REPO}/voxel_generator"

# No --overwrite: if the data is already there, stop rather than silently replace
# a dataset something downstream may already have been audited against.
srun python run_generator.py \
    --n-comp "${N_COMP}" \
    --seed 0 \
    --out-dir "${OUT}" \
    --n-train 33333 \
    --n-val 3333 \
    --n-test 3333 \
    --n-per-snr 1667 \
    --snr-ladder 20 40 60 100 150

echo
echo "finished n_comp=${N_COMP} at $(date -Iseconds)"
ls -la "${OUT}"
