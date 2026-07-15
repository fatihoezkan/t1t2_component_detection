# Voxel Simulator — Synthetic T1–T2 Correlation MRI Data Generator

Synthetic single-voxel data generator for training a **Detection Transformer (DETR)** to
quantify tissue microstructure from **T1–T2 correlation MRI**.

> Part of the thesis *"Detection Transformer for Microstructure Quantification from
> T1–T2 Correlation MRI."* This module produces the labelled `(signal → compartments)`
> pairs used to train and evaluate the model.

---

## Usage

Every voxel is the same recipe with a few knobs: **how many** voxels, **how much noise**
(relative SNR *or* absolute sigma), and the **T1/T2 ranges**. What you always get:
random compartments with **T1 > T2**, `n_comp ∈ {1,2,3}` at `0.2 / 0.6 / 0.2`, Dirichlet
weights (min 5 %), and a signed 64-point Gaussian-noisy signal.

**CLI — build the whole train/val/test + noise-ladder family:**
```bash
python run_generator.py --out-dir output/data                    # full, train SNR ∈ [30,150]
python run_generator.py --out-dir output/data --noise-sigma 0.1  # fixed absolute noise instead
python run_generator.py --help                                   # every flag (sizes, ranges, ladders)
```

**Library — one voxel.** `GeneratedVoxel.spec` is the answer (target), `.signal` is the input:
```python
from voxel_simulator.generate import generate_voxel
from voxel_simulator.protocol import load_protocol

v = generate_voxel(0, master_seed=0, protocol=load_protocol(), noise_sigma=0.1)
v.spec.n_comp, v.spec.t1, v.spec.t2, v.spec.w   # target compartments (what the model finds)
v.signal                                        # (64,) noisy signal (what the model sees)
```

**Library — a table of N voxels** (one row each; columns in the "Dataset format" section):
```python
from voxel_simulator.generate import generate_dataset, generate_dataset_family, DatasetFamilyConfig

df = generate_dataset(100_000, noise_sigma=0.1)          # or: snr_min=30, snr_max=150
generate_dataset_family(DatasetFamilyConfig(out_dir="output/data"))   # the full family, to disk
```

**Knobs** (same names on the CLI and the functions):
| Knob | Meaning | Default |
|------|---------|---------|
| `n_train / n_val / n_test` | dataset sizes | 1M / 100k / 100k |
| `snr_min, snr_max` | relative noise (used when `noise_sigma` is None) | 30, 150 |
| `noise_sigma` | absolute Gaussian std (e.g. 0.1, 0.2); overrides SNR | None |
| `t1_range, t2_range` | random-compartment ranges (ms), always `T2 < T1` | [50,4000], [5,3000] |

**Visual tour:** `jupyter notebook notebooks/05_new_system_visualization.ipynb`
**Tests:** `python -m pytest tests/ -q`

Disjoint master seeds per split give **leakage-free** train/val/test. ≈ 1 M voxels/min.

---

## 1. Background — why this exists

### The microstructure problem
A single voxel in an MR image is not a single tissue. It is a **mixture of
microscopic water pools** — myelin water, intra-/extra-cellular water, cerebrospinal
fluid — each relaxing with its own longitudinal time **T1** and transverse time **T2**.
A conventional scan reports one number per voxel and hides this mixture.

**Correlation MRI** instead samples the voxel on a 2-D grid of inversion times (TI) and
echo times (TE), producing a multi-contrast signal whose decay encodes the underlying
**(T1, T2) spectrum** of the compartments inside the voxel
[Benjamini & Basser 2020; Kim et al. 2020]. Recovering that spectrum is a classic
**ill-posed inverse problem**: small noise in the signal produces large changes in the
estimated compartments [Whittall & MacKay 1989; multi-component T2 relaxometry review].

### From inverse problem to set prediction
Classical solvers (non-negative least squares on a (T1, T2) dictionary) are unstable
under noise. The thesis reframes the task: each compartment is an **object to be
detected** in (T1, T2) space, and a **Detection Transformer**
[Carion et al. 2020] predicts the *set* of compartments
`{(T1_c, T2_c, w_c)}` directly from the signal, using **Hungarian (bipartite)
matching** so the prediction is permutation-invariant. The model architecture follows
prior work by Schlund et al. and the accompanying
`correlation-imaging-detr_t1t2` repository.

To train such a model you need millions of voxels with **known** compartments — which no
real scan can provide. **This module simulates them from first-principles MR physics.**

---

## 2. The forward model (physics)

For each of the 64 protocol points `p = (TI_p, TE_p)`, the noise-free signal of a voxel
with compartments `c` is the **inversion-recovery multi-echo (IR-MSE)** equation:

```
S_p = M0 · Σ_c  w_c · (1 − 2·exp(−TI_p / T1_c) + exp(−TR / T1_c)) · exp(−TE_p / T2_c)
```

| Term | Meaning |
|------|---------|
| `1 − 2·exp(−TI/T1) + exp(−TR/T1)` | **inversion recovery** along T1 (signed; crosses zero near `TI ≈ T1·ln 2`) |
| `exp(−TE/T2)` | **spin-echo / T2 decay** |
| `w_c`, `Σ w_c = 1` | compartment signal fractions (partial volumes) |
| `M0 = 1` | fixed overall amplitude |

The model assumes **ideal 180° inversion and refocusing** and ignores exchange,
magnetization transfer, B0/B1 inhomogeneity, and flow — standard simplifications for a
first-generation simulator [Stanisz et al. 2005; Kim et al. 2020].

Because the equation is **linear in the compartments**, a multi-pool voxel is exactly the
weighted sum of single-pool signals — a property the test-suite verifies bit-for-bit.

---

## 3. The acquisition protocol

The protocol is read from `data/ti_te_dict.mat` and used **exactly as stored**, in the
scanner's acquisition order — never re-sorted or recombined. This matters: the DETR
input is a length-64 vector whose position *p* must mean the same `(TI_p, TE_p)` at
training time and at inference on real scans.

| Property | Value |
|----------|-------|
| Measurements | **64** (ordered) |
| Unique TI | **8**, log-spaced ≈ 50 → 2000 ms (ratio ≈ 1.69) |
| Unique TE | **8**, log-spaced ≈ 4.5 → 150 ms (ratio ≈ 1.66) |
| TR | **20 000 ms** (≫ longest T1 ⇒ near-full recovery, no T1-saturation bias) |

The **log-spacing** is deliberate: brain relaxation times span more than an order of
magnitude, and log-spaced sampling gives uniform sensitivity across all T1/T2 scales —
standard practice in multi-component relaxometry [Whittall & MacKay 1989]. The
**shuffled acquisition order** spreads scanner drift and subject motion evenly across the
encoding, improving robustness of the fit.

---

## 4. Compartment sampler

The production generator is tissue-agnostic. A compartment is just a random `(T1, T2, w)`
target with `T1 > T2`; tissue prototypes live only in notebook experiments.

Per voxel the sampler draws:
- **n_comp ∈ {1, 2, 3}** with probabilities **0.2 / 0.6 / 0.2** (most brain voxels are
  2–3 pools; mono- and tri-pool are rarer);
- **T1, T2** log-uniformly across the configured ranges, with `T2 < T1`;
- **weights** from a Dirichlet(1) with an enforced **minimum 5 %** so no "ghost"
example: 
[0.72, 0.20, 0.08]
[0.33, 0.34, 0.33]
[0.95, 0.03, 0.02]
  compartment is invisible in the signal yet present in the label;
- **SNR ∈ [30, 150]** uniform.

---

## 5. Noise model — why Gaussian

The simulator keeps the signed inversion-recovery signal, including negative samples near
the zero crossing. Magnitude noise models such as Rician would rectify those values, so the
production generator uses additive Gaussian noise only:

```text
σ = max(|S_clean|) / SNR

S_noisy_p = S_clean_p + n_p

n_p ~ N(0, σ)
```

---

## 6. Dataset format

One **row = one voxel**, written to **Parquet** (`float32` signals, ≈ 0.46 KB/voxel).
81 columns in three blocks:

| Block | Columns | Description |
|-------|---------|-------------|
| **Metadata** | `voxel_id, snr, sigma, n_comp` | reproducibility + acquisition quality |
| **Ground truth** | `T1_i, T2_i, w_i` for `i = 1..3` | the DETR target set; **unused slots are `NaN`-padded** |
| **Model input** | `S_1 … S_64` | signed Gaussian-noisy signal in scanner order |

**NaN padding** (not zero) marks absent compartments — zero would imply a spurious
"T1 = 0 ms" tissue. DETR reads the NaN slots as *no-object* queries.

Reading one voxel as a `(signal, targets)` pair:

```python
signal  = row[[f"S_{p+1}"  for p in range(64)]].to_numpy(np.float32)   # (64,)
n       = int(row.n_comp)
targets = np.stack([                                                   # (n_comp, 3)
    row[[f"T1_{i+1}" for i in range(n)]].to_numpy(np.float32),
    row[[f"T2_{i+1}" for i in range(n)]].to_numpy(np.float32),
    row[[f"w_{i+1}"  for i in range(n)]].to_numpy(np.float32),
], axis=1)
```

---

## 7. Reproducibility

Every voxel is fully reproducible from `(master_seed, voxel_id)`:

- **Parameter stream** — `default_rng(voxel_seed(master_seed, voxel_id))` draws n_comp,
  (T1, T2), weights, and SNR.
- **Noise stream** — a separate `SeedSequence([master_seed, voxel_id, 7919])`. The
  hashed SeedSequence guarantees the noise stream is statistically **independent** of the
  parameter stream.

The pair `(master_seed, voxel_id)` is enough to regenerate any voxel.

---

## 8. Project layout

```
voxel_generator/
├── run_generator.py                     # the single CLI entry point (run with --help)
├── data/ti_te_dict.mat                  # fixed scanner protocol (8 TI × 8 TE, TR=20000)
├── src/voxel_simulator/
│   ├── protocol.py                      # load + freeze acquisition order
│   ├── physics.py                       # IR-MSE forward model (the equation in §2)
│   ├── noise.py                         # Gaussian signed-signal noise (§5)
│   ├── sampler.py                       # per-voxel parameter sampling (§4)
│   └── generate.py                      # voxel → dataset → train/val/test family (§6)
├── tests/                               # forward-model, noise, and sampler tests
├── notebooks/05_new_system_visualization.ipynb  # visual tour: compartments, signal images, noise
└── output/                              # generated .parquet (gitignored)
```

---

## 9. Validation checklist (all passing)

- Forward equation matches hand-computed values (`rtol 1e-12`).
- Multi-compartment signal = weighted sum of single-compartment signals (linearity).
- All signals finite.
- Weights of present compartments sum to 1.0 exactly; absent slots are NaN.
- Seeded noise is reproducible and independent of the parameter stream.
- Invalid T1/T2/weights/SNR are rejected.

---

## References

1. **Carion, N., Massa, F., Synnaeve, G., Usunier, N., Kirillov, A., & Zagoruyko, S.** (2020).
   *End-to-End Object Detection with Transformers (DETR).* ECCV. — set prediction +
   Hungarian matching backbone of the model.
2. **Schlund, S. et al.** — prior DETR-for-correlation-spectra work the model architecture
   builds on (`correlation-imaging-detr_t1t2`).
3. **Kim, D., Kim, J. W., Wisnowski, J. L., et al.** (2020). *Multidimensional correlation
   spectroscopic imaging of exponential decays: from theoretical principles to in vivo
   human applications.* NMR in Biomedicine. — T1–T2 correlation acquisition & inversion.
4. **Benjamini, D., & Basser, P. J.** (2020). *Multidimensional correlation MRI.* NMR in
   Biomedicine — review of the inverse problem.
5. **Slator, P. J., et al.** (2021). *Combined diffusion-relaxometry microstructure
   imaging: current status and future prospects.* Magnetic Resonance in Medicine.
6. **Stanisz, G. J., et al.** (2005). *T1, T2 relaxation and magnetization transfer in
   tissue at 3 T.* Magnetic Resonance in Medicine — 3 T tissue relaxation values.
7. **Wansapura, J. P., et al.** (1999). *NMR relaxation times in the human brain at 3.0 T.*
   JMRI — GM/WM T1/T2 reference values.
8. **MacKay, A., Whittall, K., et al.** (1994). *In vivo visualization of myelin water in
   brain by magnetic resonance.* Magnetic Resonance in Medicine — myelin-water short-T2 pool.
9. **Whittall, K. P., & MacKay, A. L.** (1989). *Quantitative interpretation of NMR
   relaxation data.* Journal of Magnetic Resonance — multi-exponential / NNLS inversion.
10. **Gudbjartsson, H., & Patz, S.** (1995). *The Rician distribution of noisy MRI data.*
    Magnetic Resonance in Medicine — magnitude-MRI noise statistics.
