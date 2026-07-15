# Running `t1t2_component_detection` on the THI Slurm cluster

The folder is self-contained (it vendors its own data generator and the protocol), so the workflow
is: **copy it up → set up the env → generate data → audit it → smoke the GPU → train.**

The audit is a **gate**, not a formality: a million voxels of subtly wrong data costs far more to
discover during training than the few minutes it takes to check first. `train.slurm` refuses to
start on a dataset that has not passed.

## 0. Copy the project up

Requires the THI VPN. SSH runs on **port 7070**.

```bash
# from your Mac. Exclude only the ROOT data/ and results/ — a bare 'data/' pattern would also
# match voxel_generator/data/, which holds ti_te_dict.mat, the protocol the code cannot run
# without. (That exact bug shipped once.)
rsync -av -e 'ssh -p 7070' \
      --exclude '/data/' --exclude '/results/' \
      --exclude '__pycache__' --exclude '.pytest_cache' --exclude '.venv' \
      ~/Desktop/Thesis/t1t2_component_detection/ \
      <THI_USERNAME>@<THI_HOST>:/home/<THI_USERNAME>/t1t2_component_detection/
```

Verify the protocol made it — everything downstream is meaningless without it:

```bash
ls -l /home/<THI_USERNAME>/t1t2_component_detection/voxel_generator/data/ti_te_dict.mat
```

⚠️ Keep the project and its data on **shared home**. `/fast_storage` is **node-local**: data
written there by the generation job is invisible to the training job on another node.

## 1. One-time environment setup (login node)

```bash
cd /home/<THI_USERNAME>/t1t2_component_detection
bash slurm/setup_env.sh
```

Python 3.10 venv + `torch==2.6.0+cu124` (CUDA 12.4) + `numpy==1.26.4`. Torch is installed **first**
from the CUDA index and is deliberately absent from `requirements.txt`, so nothing afterwards can
quietly swap it for a CPU build. The script verifies what actually landed and fails if not.

`cuda_available=False` on the login node is expected — it has no GPU. `smoke.slurm` proves CUDA.

## 2. Generate the balanced dataset (CPU array, ~2 h)

```bash
sbatch slurm/gen_data.slurm         # array 1-4, one task per compartment count
```

Writes `data/full_1to4/n1..n4/` — 250k train per count (**exactly 1M**), 25k val, 25k test, 12.5k
per fixed-SNR rung, plus a `manifest.json` per family (seeds, protocol SHA-256, git commit,
dependency versions). Files are written atomically and existing output is **not** overwritten.

## 3. Audit the data — THE GATE (CPU, ~4 h)

```bash
sbatch slurm/audit.slurm
```

Runs `notebooks/full_data_audit.ipynb` headless and **fails the job** if any hard check fails.
Then **read it yourself**:

```
results/data_audit/full_1to4/audit_summary.json
results/data_audit/full_1to4/figures/*.png
```

Two figures are not pass/fail but decide how the thesis reports results:
- **coverage.png** — the (T1,T2) joint. Neither marginal is log-uniform; say so in the thesis.
- **identifiability.png** — how often the smallest compartment sits below the noise floor. That
  fraction is the *ceiling* on count accuracy at each n. A low per-n number there is a property of
  the data, not a failure of the model.

## 4. GPU smoke (~minutes)

```bash
sbatch slurm/smoke.slurm
```

Two epochs on CUDA + a checkpoint/resume test + a **wall-time projection** from the measured
sec/epoch. Do not skip this: whether 200 epochs of 1M voxels fits 24 h is a measurement, not a
guess — and guessing about this code's performance has already been wrong once (the "Hungarian
matching is the bottleneck" claim was measured false).

## 5. Train (GPU, up to 24 h)

```bash
sbatch slurm/train.slurm                          # configs/cluster.yaml
sbatch slurm/train.slurm configs/scale_200k.yaml  # a data-scaling arm

squeue -u $USER
tail -f slurm/logs/train_*.out
```

`epochs: 200` is an **upper bound**; early stopping (patience 10) decides when it ends, and the
evaluated model is `best.pt`, not the final epoch.

Hit the 24 h wall? Just resubmit — the run resumes from `last.pt` with its optimizer, epoch,
history and patience state. Resume **refuses** to continue if the config or dataset changed, so a
resubmit cannot silently blend two experiments.

## Resources

| Job | partition/qos/account | GPU | CPU | RAM | time |
|---|---|---|---|---|---|
| `gen_data.slurm` (array 1–4) | `debugging` | — | 1/task | 16 G | 2 h |
| `audit.slurm` | `debugging` | — | 4 | 32 G | 4 h |
| `smoke.slurm` | `debugging` | `gpu:1g.10gb:1` | 4 | 32 G | 2 h |
| `train.slurm` | `advance` | `gpu:nvidia_a100_80gb_pcie_3g.39gb:1` | 8 | 32 G | 24 h |

## Notes

- `device` is `null` in the configs → auto-detects **cuda** on the GPU node.
- Every job sources `slurm/_common.sh`, which finds the repo from the script's own path (**not**
  `$SLURM_SUBMIT_DIR` — a job that only works when submitted from one directory breaks later for a
  reason nobody remembers), activates the venv, checks the protocol is present, and logs host,
  git commit + dirty flag, protocol checksum, python/numpy/torch versions and the GPU name.
- The 8 CPUs on the training job are for dataloading. The Hungarian matching was measured at
  **5.4 µs/voxel** (~5 s/epoch at 1M) — it is not the bottleneck the older note claimed.
- Override the repo location with `T1T2_REPO`, the venv with `T1T2_VENV`, the interpreter with
  `T1T2_PYTHON`.
