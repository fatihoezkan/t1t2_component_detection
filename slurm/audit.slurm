#!/usr/bin/env bash
#SBATCH --job-name=t1t2_audit
#SBATCH --partition=debugging
#SBATCH --qos=debugging
#SBATCH --account=debugging
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=slurm/logs/audit_%j.out
#SBATCH --error=slurm/logs/audit_%j.out
#
# THE GATE. Runs notebooks/full_data_audit.ipynb headless against the generated dataset and fails
# the job if any hard check fails. No training is submitted until this is green AND a human has
# read the figures in results/data_audit/<name>/.
#
#     sbatch slurm/audit.slurm                    # audits data/baseline_100k
#     AUDIT_DATA_ROOT=data/dev_1to4 sbatch slurm/audit.slurm
#
# A dataset that fails here is cheap to regenerate. A training run built on one that should have
# failed is not.

source "/home/fao8402/t1t2_component_detection/slurm/_common.sh"

export AUDIT_DATA_ROOT="${AUDIT_DATA_ROOT:-data/baseline_100k}"
export AUDIT_EXPECT_TRAIN="${AUDIT_EXPECT_TRAIN:-33333}"   # exact per-n size; balance alone is not enough
# On the cluster "which code made this data?" must have an answer, so a dirty worktree is a hard
# failure here (locally it is only a warning — dev data is regenerated constantly from a dirty tree).
export AUDIT_REQUIRE_CLEAN=1

echo "auditing ${AUDIT_DATA_ROOT} (expecting ${AUDIT_EXPECT_TRAIN} train voxels per n)"
cd "${REPO}/notebooks"

OUT="${REPO}/results/data_audit/$(basename "${AUDIT_DATA_ROOT}")"
SUMMARY="${OUT}/audit_summary.json"
# A failed notebook must not be allowed to inherit a previous green verdict.
rm -f "${SUMMARY}"

# --allow-errors so the notebook is always written, with the failure visible in it. Without it,
# nbconvert discards the run on the first error — leaving the notebook showing some older, greener
# execution, which is the worst possible artifact for a gate. The verdict comes from the summary
# below, not from nbconvert's exit code.
srun python -m nbconvert --to notebook --execute --inplace full_data_audit.ipynb \
    --ExecutePreprocessor.timeout=10800 --allow-errors

if [[ ! -f "${SUMMARY}" ]]; then
    echo "FATAL: the audit did not finish — no ${SUMMARY}. Read the notebook for the traceback." >&2
    exit 1
fi

python -m json.tool "${SUMMARY}"
if ! python -c "import json,sys; sys.exit(0 if json.load(open('${SUMMARY}'))['passed'] else 1)"; then
    echo
    echo "AUDIT FAILED -> do not train on $(basename "${AUDIT_DATA_ROOT}")." >&2
    echo "The notebook shows which checks failed: ${REPO}/notebooks/full_data_audit.ipynb" >&2
    exit 1
fi

echo
echo "AUDIT PASSED -> $(basename "${AUDIT_DATA_ROOT}")"
echo "Now read the figures before submitting training — passing is necessary, not sufficient:"
echo "  ${OUT}/figures/"
