#!/usr/bin/env bash
#SBATCH --job-name=t1t2_gen_3500_500
#SBATCH --partition=debugging
#SBATCH --qos=debugging
#SBATCH --account=debugging
#SBATCH --array=1-3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=slurm/logs/gen_3500_500_%A_%a.out
#SBATCH --error=slurm/logs/gen_3500_500_%A_%a.out

source "/home/fao8402/t1t2_component_detection/slurm/_common.sh"

N_COMP="${SLURM_ARRAY_TASK_ID}"
OUT="${REPO}/data/t1_3500_t2_500_100k/n${N_COMP}"

cd "${REPO}/voxel_generator"
srun python run_generator.py \
    --n-comp "${N_COMP}" \
    --seed 3500500 \
    --out-dir "${OUT}" \
    --n-train 33333 \
    --n-val 3333 \
    --n-test 3333 \
    --n-per-snr 1667 \
    --snr-min 30 \
    --snr-max 150 \
    --snr-ladder 20 40 60 100 150 \
    --t1-min 50 \
    --t1-max 3500 \
    --t2-min 5 \
    --t2-max 500
