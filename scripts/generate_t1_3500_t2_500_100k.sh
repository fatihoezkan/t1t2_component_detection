#!/usr/bin/env bash
# Generate the new bounded-relaxation experiment without touching any existing dataset.
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_ROOT="${PROJECT_ROOT}/data/t1_3500_t2_500_100k"

for N_COMP in 1 2 3; do
    OUT="${OUT_ROOT}/n${N_COMP}"
    if [[ -f "${OUT}/manifest.json" ]]; then
        echo "complete manifest exists; keeping ${OUT}"
        continue
    fi
    if [[ -d "${OUT}" ]] && find "${OUT}" -mindepth 1 -print -quit | grep -q .; then
        echo "refusing partial non-empty directory without a manifest: ${OUT}" >&2
        exit 1
    fi
    python3 "${PROJECT_ROOT}/voxel_generator/run_generator.py" \
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
done

echo "generated new data under ${OUT_ROOT}"
