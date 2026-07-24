# t1t2_component_detection

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
t1t2_component_detection/
├── voxel_generator/     # vendored snapshot of the data generator (see PROVENANCE.txt)
├── data/                # generated parquet splits (gitignored)
│   ├── dev_1to4/        # local data-analysis/smoke family
│   └── baseline_100k/   # generated on cluster: n1..n3, 99,999 train voxels
├── src/t1t2/            # model, loss, train, evaluation, and experiment package
├── configs/             # experiment YAMLs — a run is fully described by one of these
├── slurm/               # cluster data generation and baseline training
├── results/             # run artifacts: metrics, figures, checkpoints (gitignored)
└── tests/               # architecture, data, training/resume, and evaluation checks
```

## Status (current milestone)

The training pipeline is ready. The first cluster baseline is
deliberately narrow: **64 inputs, n_comp=1..3, 99,999 balanced train voxels**, the inherited DETR
architecture, constant learning rate, early stopping, and no auxiliary or physics-consistency
loss. Validation, test and the SNR ladder bring the complete generated family to 145,002 voxels.
No full GPU training result exists yet.

## Commands

```bash
# local tests
PYTHONPATH=src python3 -m pytest tests/ -q

# small local baseline (uses data/dev_1to4/n1..n3)
PYTHONPATH=src python3 -m t1t2.experiment --config configs/baseline.yaml
```

For the cluster sequence and exact commands, see [`slurm/README.md`](slurm/README.md).

## Design notes

- **Fixed at 64 inputs** (8 TI × 8 TE), protocol used exactly as stored in scanner order —
  input position *p* must mean the same (TI_p, TE_p) at train and inference time.
- **Signed data → Gaussian noise.** The inversion-recovery signal genuinely goes negative;
  the generator keeps the sign and adds additive Gaussian noise (not Rician). A future
  signal-consistency loss therefore uses plain MSE, not a Rician term.
- **log-minmax T1/T2 normalization** by default: relaxation times span a decade, so log-space
  spreads them evenly and lets the loss weights all sit near 1.0.
- Compartment **count** is read off as the number of queries with existence > 0.5.
- The first baseline excludes `n=4`; adding it is a separate stress-test experiment, not a silent
  expansion of the required thesis scope.
