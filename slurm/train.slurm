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
# Baseline training. Resubmitting resumes from the last checkpoint.

source "/home/fao8402/t1t2_component_detection/slurm/_common.sh"

CFG="${1:-configs/cluster.yaml}"

[[ -f "${CFG}" ]] || { echo "missing config: ${CFG}" >&2; exit 1; }

echo "training ${CFG}"
srun python -m t1t2.experiment --config "${CFG}"
