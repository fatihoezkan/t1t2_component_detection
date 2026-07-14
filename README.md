# t1t2_training

A **from-scratch** Detection Transformer (DETR) training codebase for the thesis
*"Detection Transformer for Microstructure Quantification from T1–T2 Correlation MRI."*
Self-contained on purpose — it vendors its own copy of the data generator, so the whole
folder can be `rsync`-ed to the HPC cluster as one unit.

## The idea

One MRI voxel isn't one tissue — it's a blend of microscopic water pools (myelin water,
grey/white matter, CSF), each relaxing at its own T1 and T2. A correlation scan samples the
voxel across a grid of 8 inversion × 8 echo times, and the resulting **64-number signal**
secretly encodes the mix. Pulling the compartments back out is a classically unstable inverse
problem (NNLS on a (T1,T2) dictionary). We reframe it as **detection**: each compartment is an
object to find in (T1, T2) space, a DETR predicts a *set* `{(T1, T2, weight)}` + existence, and
a **Hungarian-matching** loss scores that set order-independently.

## Layout

```
t1t2_training/
├── voxel_generator/     # vendored snapshot of the data generator (see PROVENANCE.txt)
├── data/                # generated parquet splits (gitignored)
│   └── dev/             #   small split for local smoke: train/val/test + fixed-SNR ladder
├── src/t1t2/            # the model + training package (this milestone: config/device/data/model/loss)
├── configs/             # experiment YAMLs — a run is fully described by one of these
├── slurm/               # cluster submit scripts (later milestone)
├── results/             # run artifacts: metrics, figures, checkpoints (gitignored)
└── tests/               # architecture smoke tests (shapes + gradient; no training)
```

## Status (current milestone)

Built and verified: `config` (YAML experiments), `device` (cuda>mps>cpu), `data` (Parquet
loader + swappable T1/T2 normalization), `model` (`T1T2DETR`), `loss` (`HungarianLoss`),
`physics` (differentiable IR-MSE forward), and the `train` / `eval` / `experiment` pipeline. A
small dev dataset is generated under `data/dev/`. **No full training run yet** — that (on the
GPU cluster) and the Slurm scripts are the next milestones.

## Commands

```bash
# regenerate the small dev dataset (from the vendored generator)
cd voxel_generator && PYTHONPATH=src python run_generator.py \
    --out-dir ../data/dev --n-train 50000 --n-val 10000 --n-test 10000 --n-per-snr 5000

# architecture smoke tests (shapes + gradient flow — does not train)
cd t1t2_training && PYTHONPATH=src python -m pytest tests/ -q
```

## Design notes

- **Fixed at 64 inputs** (8 TI × 8 TE), protocol used exactly as stored in scanner order —
  input position *p* must mean the same (TI_p, TE_p) at train and inference time.
- **Signed data → Gaussian noise.** The inversion-recovery signal genuinely goes negative;
  the generator keeps the sign and adds additive Gaussian noise (not Rician). A future
  signal-consistency loss therefore uses plain MSE, not a Rician term.
- **log-minmax T1/T2 normalization** by default: relaxation times span a decade, so log-space
  spreads them evenly and lets the loss weights all sit near 1.0.
- Compartment **count** is read off as the number of queries with existence > 0.5.
