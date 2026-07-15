#!/usr/bin/env bash
# One-time cluster environment setup. Run once on a login node, from anywhere:
#
#     bash slurm/setup_env.sh
#
# Creates .venv, installs a CUDA-matched torch, then the rest of the requirements.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO}"
VENV="${T1T2_VENV:-${REPO}/.venv}"

PY="${T1T2_PYTHON:-python3.10}"          # THI: Python 3.10
TORCH_VERSION="2.6.0"
CUDA_TAG="cu124"                         # THI: CUDA 12.4

if ! command -v "${PY}" >/dev/null 2>&1; then
    echo "FATAL: ${PY} not found. Load a Python 3.10 module first, or set T1T2_PYTHON." >&2
    exit 1
fi

echo "==> creating venv at ${VENV} with $(${PY} --version)"
"${PY}" -m venv "${VENV}"
# shellcheck disable=SC1091
source "${VENV}/bin/activate"
python -m pip install --upgrade pip wheel

# Torch FIRST, from the CUDA-matched index. The order matters: requirements.txt deliberately does
# not list torch, so nothing afterwards can quietly swap this build for a default one.
echo "==> installing torch==${TORCH_VERSION}+${CUDA_TAG}"
python -m pip install "torch==${TORCH_VERSION}" --index-url "https://download.pytorch.org/whl/${CUDA_TAG}"

echo "==> installing the rest"
python -m pip install -r requirements.txt

# Verify what we ended up with, not what we asked for.
echo "==> verifying"
python - <<PY
import platform, sys
import numpy, pandas, pyarrow, torch

print(f"python : {platform.python_version()}")
print(f"numpy  : {numpy.__version__}")
print(f"pandas : {pandas.__version__}  pyarrow: {pyarrow.__version__}")
print(f"torch  : {torch.__version__}  (built against cuda {torch.version.cuda})")
print(f"cuda   : available={torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"gpu    : {torch.cuda.get_device_name(0)}")

problems = []
if not torch.__version__.startswith("${TORCH_VERSION}"):
    problems.append(f"torch is {torch.__version__}, expected ${TORCH_VERSION}")
if torch.version.cuda is None:
    problems.append("torch has no CUDA build — something replaced the cu124 wheel")
if numpy.__version__ != "1.26.4":
    problems.append(f"numpy is {numpy.__version__}, expected 1.26.4 "
                    "(the generator's reproducibility is pinned to it)")
if problems:
    print("\nFAILED:")
    for p in problems:
        print("  -", p)
    sys.exit(1)
print("\nenvironment OK")
PY

echo
echo "Done. cuda_available=False on a login node is expected — it has no GPU."
echo "slurm/smoke.slurm is what proves CUDA actually works."
