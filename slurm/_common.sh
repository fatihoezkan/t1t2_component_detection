#!/usr/bin/env bash
# Shared cluster paths.
set -euo pipefail

REPO="/home/fao8402/t1t2_component_detection"
VENV="/home/fao8402/venvs/thesis"
[[ -f "${VENV}/bin/activate" ]] || { echo "missing environment: ${VENV}" >&2; exit 1; }
# shellcheck disable=SC1091
source "${VENV}/bin/activate"
cd "${REPO}"
export PYTHONPATH="${REPO}/src:${REPO}/voxel_generator/src:${PYTHONPATH:-}"
