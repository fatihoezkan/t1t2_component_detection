# Credits & Provenance

This codebase (`t1t2_component_detection`) is a from-scratch, T1–T2-native
implementation for the bachelor thesis *"Detection Transformer for Microstructure
Quantification from T1–T2 Correlation MRI"* (Fatih Özkan; advisor: Sebastian Endt).

The Detection Transformer architecture is **adopted and adapted** from prior work,
credited below. The scientific contribution of the thesis is not the DETR
architecture itself but the reframing of the T1–T2 spectral inverse problem as set
prediction, the input-count ablation, and the real-data feasibility study.

## Builds on

- **T1T2DETR architecture — Sebastian Endt, `correlation-imaging-detr_t1t2`**
  <https://github.com/SebastianEndtTHI/correlation-imaging-detr_t1t2> (pinned @ `f5b90d4`, 2026-04-02)
  Adopted design: MLP signal-encoder → learned queries → transformer decoder →
  per-query heads (T1, T2, weight via sigmoid + a concatenated-query existence head);
  the Hungarian bipartite-matching loss structure; and the fixed 8×8 TI×TE
  acquisition protocol (`data/ti_te_dict.mat`), used exactly as stored.

- **DETR-for-microstructure idea — Johannes Schlund** (diffusion-diffusion
  correlation DETR, predecessor work): the original idea of solving microstructure
  quantification as DETR set prediction.

- **DETR — Carion et al.**, *End-to-End Object Detection with Transformers*, ECCV 2020:
  the base method (Hungarian-matched set prediction with a transformer).

- **Synthetic data generator — `voxel_simulator`** (first-principles IR-MSE forward
  model + additive Gaussian noise). Vendored from upstream `512fc66`, then extended for
  fixed-count families, reproducible split streams, paired-SNR tests, and manifests; see
  `voxel_generator/PROVENANCE.txt`.

## Original to this repo

A clean, T1–T2-native reimplementation rather than a fork:

- Config-driven experiment framework (YAML → dataclasses) covering
  config · device · data · model · loss · physics · train · eval · experiment.
- Log-minmax target normalization for the sigmoid heads (swappable normalizer).
- IR-MSE forward model implemented in both numpy and torch, **parity-tested**
  against the data generator so training targets and physics agree bit-for-bit.
- Evaluation with CSF-T2 stratification (CSF is weakly identifiable at TE_max ≈ 150 ms)
  and predicted-vs-true scatter figures.
- Cluster-first Slurm scaffolding (data generation + GPU training, resume-safe).

**Scientific contribution of the thesis:** the T1–T2 reframing of the inverse
problem, the input-count ablation (how few inputs still give reasonable results,
target 64), and the real-data (in-vivo 64-input) feasibility evaluation.

## Fixes contributed back (not yet upstreamed)

While getting the reference repo to run, we made run/correctness fixes to
`correlation-imaging-detr_t1t2` (in `training/main.py`, `match_loss.py`,
`trainutils.py`, and `README.md`). These live locally and are **not committed**;
they can be offered back to the upstream repo as a pull request.
