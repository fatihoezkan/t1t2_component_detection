# Progress Report — `t1t2_training`

*Detection Transformer for Microstructure Quantification from T1–T2 Correlation MRI.*
Status as of 2026-07-10. This report presents what has been built so far and a detailed
comparison against the prior **diffusion** DETR repositories the thesis descends from.

---

## 1. Where we are

A fresh, self-contained training codebase (`t1t2_training/`) has been built **from scratch**
and validated end-to-end on a small local dataset. **No full training has been run yet** — that
is cluster work — but every component is in place and tested (**9/9 tests green**), and a
short local training demo confirms the loop learns.

| Component | State |
|---|---|
| New self-contained folder (`t1t2_training/`) | ✅ built |
| Vendored data generator (`voxel_generator/`, pinned to upstream `512fc66`) | ✅ tests 15/15 |
| Small dev dataset (`data/dev/`, 95k voxels + fixed-SNR ladder) | ✅ sanity-checked |
| Architecture: `config, device, data, model, loss` | ✅ smoke 5/5 |
| Pipeline: `physics, train, eval, experiment` | ✅ pipeline 4/4 |
| Demo notebook (`notebooks/demo.ipynb`), run locally | ✅ executed |
| Full cluster training / ablation / real-data eval | ⏳ next milestones |

---

## 2. What was built

**Self-contained folder** that `rsync`s to the cluster as one unit:

```
t1t2_training/
├── voxel_generator/     # vendored snapshot of the generator (PROVENANCE.txt pins the commit)
├── data/dev/            # train/val/test + fixed-SNR ladder (20/40/60/100/150), 95k voxels
├── src/t1t2/            # config · device · data · model · loss · physics · train · eval · experiment
├── configs/baseline.yaml
├── tests/               # 9 tests (architecture smoke + pipeline)
└── notebooks/demo.ipynb # end-to-end local demo
```

**The model (`model.py`).** `T1T2DETR`: a 4-layer MLP encoder compresses the 64-point signal
into a 256-wide feature; 10 learned query embeddings attend to it through a 4-layer
`TransformerDecoder`; per-query heads emit `(T1, T2, weight)` via sigmoid plus a
concatenated-query existence logit. Output `(B, 10, 4)`.

**The loss (`loss.py`).** `HungarianLoss`: an all-pairs cost (weight-scaled T1+T2 MSE + weight
MSE + existence cost), solved per voxel with `scipy.linear_sum_assignment` using `n_comp` to
drop padded targets, then matched-pair regression + a positive-balanced BCE existence term.

**The data layer (`data.py`).** Reads the generator's Parquet directly, maps T1/T2 into `[0,1]`
with a swappable `TargetNormalizer` (default **log-minmax** — relaxation times span a decade),
and pads absent compartments with inert zeros gated by `n_comp`.

**Physics (`physics.py`).** The IR-MSE forward model re-implemented in numpy **and**
differentiable torch, **parity-tested against the generator to `rtol 1e-12`** — the hook for a
future signal-consistency loss and a guarantee that training targets and the physics agree.

**Training + evaluation (`train.py`, `eval.py`, `experiment.py`).** Device-agnostic loop
(cuda>mps>cpu) with checkpoint/resume; evaluation reports count accuracy, per-parameter error
(ms + relative), a **CSF-T2 breakout** (weakly identifiable at TE_max≈150 ms), and the
predicted-vs-true (T1,T2) scatter. One CLI runs train→eval.

**The physics, in one line:** `S_p = Σ_c w_c·(1 − 2e^{−TI_p/T1_c} + e^{−TR/T1_c})·e^{−TE_p/T2_c}` —
signed (goes negative near the inversion zero-crossing), so the noise is additive **Gaussian**,
not Rician.

---

## 3. Detailed comparison with the diffusion repos

The thesis descends from the **diffusion** DETR of Johannes Schlund (continued by Marcos),
`git_repo/Diffusion-DETR[-main]/`, via an intermediate T1–T2 port
(`git_repo/correlation-imaging-detr_t1t2/`, Sebastian). Below is what carried over, what
changed, and what is new.

### 3.1 Side-by-side

| Aspect | Diffusion-DETR (Johannes/Marcos) | **This thesis — `t1t2_training`** |
|---|---|---|
| **Physical signal** | diffusion tensor: `S = exp(−b·rᵀDr)`, per-compartment MD/FA → eigenvalues → direction + rotation → tensor D | **IR-MSE relaxation:** `S = Σ w·(1−2e^{−TI/T1}+e^{−TR/T1})·e^{−TE/T2}` |
| **Sign of signal** | strictly **positive** (monotone decay) | **signed** (inversion recovery crosses zero) |
| **Noise** | additive Gaussian (`noise_level` absolute, e.g. 0.01) | additive **Gaussian**, signed-preserving (SNR **or** absolute σ) — Rician would be *wrong* here |
| **# inputs (measurements)** | **166** directions (`diff_dirs_166`); Johannes' original >300 | **64** (8 TI × 8 TE) — clinically feasible scan time |
| **Outputs per query** | **7**: MD, FA, dir x, dir y, dir z, weight, existence | **4**: T1, T2, weight, existence |
| **Regression targets** | MD, FA, **3-vector direction** | **T1, T2** (no direction) |
| **Loss terms** | MD (÷4), FA, **direction** (cosine / normalized-angular), weight (MSE **or MAPE**), existence (**BCE or focal**) | weight-scaled T1+T2 MSE, weight MSE, existence **BCE** (pos-balanced) |
| **Encoder** | MLP + **optional Transformer self-attention** | MLP only (kept minimal) |
| **Target scaling** | MD divided by 4 in the cost | **log-minmax** T1/T2 → `[0,1]` |
| **Data format** | **CSV** (`mri_sigX`, `MD_comp*`, `FA_comp*`, `dir*`, `w*`, `n_comp`) | **Parquet** (`S_1..64`, `T1/T2/w_i` NaN-padded, `n_comp`) |
| **Config** | argparse flags | **YAML experiment configs** (a run = one file) |

### 3.2 What we KEPT (faithful to the prior work)

The whole DETR skeleton is inherited, because that is precisely the method the thesis builds on:
- **Set prediction + Hungarian bipartite matching** for permutation invariance.
- **Learned query embeddings → `TransformerDecoder`** producing a fixed set of guesses.
- **Concatenated-query existence head** + **positive-balanced BCE**, and compartment **count =
  #(existence > 0.5)**.
- **Weight-scaled matching cost** (dominant pools cost more to get wrong), **minimum-5% Dirichlet
  weights**, and `n_comp`-gated padding.

### 3.3 What we CHANGED / SIMPLIFIED

- **Signal model swapped** diffusion tensor → **T1–T2 inversion-recovery multi-echo**. This is
  the core scientific change; everything downstream follows from it.
- **7 outputs → 4.** The three direction coordinates disappear (relaxation has no orientation),
  and MD/FA are replaced by T1/T2. Fewer outputs is the thesis's central bet: *less to extract ⇒
  hope to need fewer inputs ⇒ clinically feasible.*
- **Loss stripped to the essentials.** All the diffusion-specific machinery — the cosine /
  normalized-angular **direction loss**, the **focal-loss** existence option, the **MAPE** weight
  option — is removed. What remains is the minimal 4-term loss, with O(1) weights made possible by
  log-normalizing T1/T2 (no powers-of-ten scale hacks).
- **166/300+ inputs → 64**, fixed to the real-data protocol and used in exact scanner order.
- **CSV → Parquet**, with deliberate **NaN padding** (0 would imply a spurious "T1=0" tissue).
- **Encoder kept minimal** (no optional self-attention block) — matches the intermediate T1–T2
  port and keeps the 64→feature map simple.

### 3.4 What we ADDED (new in this thesis)

- **A differentiable torch forward model**, parity-checked against the numpy generator — the hook
  for a future **signal-consistency loss** (resynthesize the signal from a prediction; plain MSE
  is correct here because the data is signed).
- **A config-driven experiment framework** (typed YAML → dataclasses), device-agnostic training
  with checkpoint/resume, and **CSF-stratified evaluation** (T2 is weakly identifiable at
  TE_max≈150 ms, so it is reported separately rather than distorting the headline metric).
- **A test suite (9)** covering physics parity, forward/loss shapes, gradient flow, and
  train/resume/eval — none of which existed for the T1–T2 case before.

---

## 4. How to run

```bash
cd t1t2_training

# tests (architecture + pipeline)
PYTHONPATH=src python -m pytest tests/ -q                       # 9 passed

# regenerate the small dev dataset
cd voxel_generator && PYTHONPATH=src python run_generator.py \
    --out-dir ../data/dev --n-train 50000 --n-val 10000 --n-test 10000 --n-per-snr 5000 && cd ..

# the end-to-end demo notebook (protocol → signal → model → train → eval)
PYTHONPATH=src jupyter nbconvert --to notebook --execute --inplace \
    --ExecutePreprocessor.timeout=1200 notebooks/demo.ipynb

# one experiment (train + eval), once the cluster splits exist
PYTHONPATH=src python -m t1t2.experiment --config configs/baseline.yaml
```

## 5. What's next

1. **Full-scale cluster training** — generate `data/full/` (1M/100k/100k + SNR ladder) and train
   the 64-input baseline on the GPU; Slurm submit scripts + environment setup.
2. **Input-count ablation** (Thesis Goal 2) — 64 → 48 → 32 → 16, the core novel result.
3. **Real-data evaluation** on the group's one small 64-input in-vivo set.
4. **Extensions** (gated on time): signal-consistency loss, neighborhood (3×3) input.
