#!/usr/bin/env bash
#SBATCH --job-name=t1t2_train_3500_500
#SBATCH --partition=advance
#SBATCH --qos=advance
#SBATCH --account=advance
#SBATCH --gres=gpu:nvidia_a100_80gb_pcie_3g.39gb:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=slurm/logs/train_3500_500_%j.out
#SBATCH --error=slurm/logs/train_3500_500_%j.out

source "/home/fao8402/t1t2_component_detection/slurm/_common.sh"

CFG="configs/t1_3500_t2_500_weighted_long.yaml"
[[ -f "${CFG}" ]] || { echo "missing config: ${CFG}" >&2; exit 1; }
AUDIT="results/data_audit/t1_3500_t2_500_100k/audit_summary.json"
[[ -f "${AUDIT}" ]] || { echo "missing audit verdict: ${AUDIT}" >&2; exit 1; }
python -c "import json,sys; sys.exit(0 if json.load(open('${AUDIT}'))['passed'] else 1)"

srun python -m t1t2.experiment --config "${CFG}"
