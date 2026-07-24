#!/usr/bin/env bash
#SBATCH --job-name=t1t2_audit_3500_500
#SBATCH --partition=debugging
#SBATCH --qos=debugging
#SBATCH --account=debugging
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=slurm/logs/audit_3500_500_%j.out
#SBATCH --error=slurm/logs/audit_3500_500_%j.out

source "/home/fao8402/t1t2_component_detection/slurm/_common.sh"

srun python scripts/audit_t1_3500_t2_500_100k.py \
    --data-root data/t1_3500_t2_500_100k \
    --output results/data_audit/t1_3500_t2_500_100k
