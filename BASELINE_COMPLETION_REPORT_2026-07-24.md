# Baseline completion report — 2026-07-24

## Verdict

The 64-input, single-voxel, `n_comp=1..3` synthetic DETR baseline is complete as a reproducible
proof of concept. The selected checkpoint trains stably, predicts physically sensible parameters,
and now has a validation-calibrated evaluation package with quantitative metrics, noise
robustness, query analysis, and deterministic success/failure examples.

This verdict is limited to synthetic signed IR-MSE signals with additive Gaussian noise. It does
not establish real-data performance or close the simulation-to-real gap.

## Run identity

- Training set: 99,999 voxels, balanced as 33,333 each for one, two, and three compartments.
- Validation/test: 9,999 voxels each, balanced as 3,333 per compartment count.
- Model: 4,508,365 parameters, ten learned queries, four decoder layers.
- Selected checkpoint: epoch 18 (zero-based checkpoint epoch 17).
- Best validation loss during training: 0.0241917.
- Training stopped after 28 epochs.
- Dataset base seed: 0, with independent counter-based parameter/noise/SNR streams.
- Dataset audit: 39 checks, zero failures, zero warnings.
- Protocol SHA-256:
  `78ab7a82ce2c1ef91f7379a466aede3496340c271d07d13765b93ec766f9265d`.
- Checkpoint SHA-256:
  `243f4292ea3b9361c43f512dfe30351b850ce245854fe8368c9ef00d05ab3281`.

The locally available larger dataset families were not treated as a new evaluation sample. The
finalizer verified their base seed, stream IDs, split codes, physics ranges, protocol checksum,
and sequential voxel IDs, then used the exact baseline-length prefix of each split.

## Threshold selection

The existence threshold was selected on the validation split only, using a fixed 0.05–0.95 grid.
The primary objective was exact compartment-count accuracy; count MAE and existence F1 were
tie-breakers.

- Selected threshold: **0.79**.
- Validation count accuracy at 0.79: **74.13%**.
- Validation count accuracy at the previous untuned 0.50 threshold: **63.91%**.
- Frozen-threshold test count accuracy: **74.01%**.

Threshold calibration therefore improves the test headline by approximately **10.1 percentage
points** relative to the historical 0.50-threshold test result, without retraining or test-set
tuning.

## Final test results

| Metric | Result |
|---|---:|
| Exact count accuracy | 74.01% |
| Count MAE | 0.271 compartments/voxel |
| Existence precision | 96.03% |
| Existence recall | 90.17% |
| Existence F1 | 93.01% |
| Median matched relative T1 error | 9.17% |
| Median matched relative T2 error | 14.42% |
| Median matched absolute weight error | 4.44 percentage points |
| Mean matched absolute T1 error | 205.4 ms |
| Mean matched absolute T2 error | 63.6 ms |
| Mean matched absolute weight error | 8.91 percentage points |
| `T2 >= T1` prediction rate | 0.45% |
| Median predicted weight-sum deviation from one | 0.034 |
| Median predicted-signal vs observed-signal RMSE | 0.02665 |
| Median true-clean-signal vs observed-signal RMSE | 0.01198 |

Matched parameter errors can look optimistic when the model misses a compartment. The artifact
also records the same regression metrics conditioned on correct count, plus the complete count
confusion matrix, so this limitation is explicit rather than hidden.

### Results by true compartment count

| True count | Exact count accuracy | Median relative T1 | Median relative T2 | Median absolute weight |
|---:|---:|---:|---:|---:|
| 1 | 96.49% | 2.94% | 4.44% | 0.86 pp |
| 2 | 64.63% | 8.63% | 13.47% | 5.12 pp |
| 3 | 60.91% | 17.30% | 26.30% | 6.73 pp |

The baseline is excellent for single compartments but remains limited for mixtures. For
three-compartment voxels, 36.0% are predicted as two and 3.1% as one. This is the main baseline
failure mode and is consistent with the dataset audit’s identifiability result: weak or nearby
components can fall below the noise floor.

## Noise robustness

The same frozen threshold was used at every SNR. The fixed-SNR sets are paired: parameters and
standardized noise are identical across rungs, so the changes are attributable to noise amplitude.
SNR 20 is an extrapolation result because training uses SNR 30–150.

| SNR | In training range | Count accuracy | Existence F1 | Median T1 rel. | Median T2 rel. | Median weight abs. |
|---:|:---:|---:|---:|---:|---:|---:|
| 20 | No | 61.97% | 89.53% | 13.11% | 19.65% | 6.09 pp |
| 40 | Yes | 70.69% | 92.10% | 9.86% | 15.62% | 4.73 pp |
| 60 | Yes | 73.01% | 92.74% | 9.27% | 14.39% | 4.44 pp |
| 100 | Yes | 74.43% | 93.13% | 8.66% | 13.83% | 4.25 pp |
| 150 | Yes | 74.77% | 93.24% | 8.50% | 13.67% | 4.21 pp |

Performance degrades smoothly rather than collapsing under noise. Improvements largely plateau
between SNR 100 and 150, suggesting that ambiguity in the inverse problem—not measurement noise
alone—limits the current architecture/data formulation.

## What the learned queries do

The query analysis does not assume that a fixed query represents a biological tissue. It measures
activity and Hungarian matches empirically.

- Q2 and Q3 are active on approximately 57.5% and 57.9% of voxels.
- Q4 is active on 41.2% and is the dominant single-compartment query: it activates on 98.9% of
  true `n=1` voxels.
- Q7 is active on 30.9%.
- Q8 is active on only 0.33%.
- Q0, Q1, Q5, Q6, and Q9 are inactive at the calibrated threshold.

There is partial parameter specialization:

- Q2 tends toward longer components (matched median T1 1543 ms, T2 192 ms).
- Q3 tends toward shorter components (matched median T1 297 ms, T2 20 ms).
- Q4 has matched medians T1 709 ms and T2 46 ms and primarily handles single-component voxels.
- Q7 has matched medians T1 558 ms and T2 40 ms and is used mainly in multi-component voxels.

The matched distributions still overlap substantially, so the queries are not fixed tissue labels.
However, five completely inactive queries and one nearly inactive query provide a strong,
data-backed reason to run the planned 5/10/15/20 query-count ablation.

## Reproduction and artifacts

Run from the project root:

```bash
MPLCONFIGDIR=/tmp/matplotlib-baseline-final \
PYTHONPATH=src python3 -m t1t2.baseline \
  --config configs/baseline_final_100k.yaml
```

The command refuses to write into an existing result directory unless explicitly allowed. The
completed package is under `results/baseline_final_100k/` and contains:

- `baseline_summary.json`
- `metrics_test.json`
- `metrics_snr_ladder.json`
- `threshold_calibration.json`
- `query_analysis.json`
- `prediction_examples.json`
- `provenance.json`
- compressed validation/test raw query outputs
- training, calibration, confusion, per-count error, matched scatter, SNR, query-specialization,
  and success/failure figures

Verification on 2026-07-24:

```text
31 passed
```

## Next controlled experiments

1. Train query-count arms at 5, 10, 15, and 20 queries with all other settings fixed. Report
   accuracy, false positives/misses, inference time, and parameter count. The five inactive
   baseline queries make this the most immediately justified experiment.
2. Run a limited Optuna study on validation data only. Search loss weights, learning rate, weight
   decay, and possibly the existence threshold; never expose the test split to trial selection.
3. Add the differentiable IR-MSE signal-consistency loss as a clean baseline-vs-physics ablation.
4. Only then compare explicit post-processing and additional noise/separation stress tests.

Real-data evaluation remains conditional on choosing a stable synthetic configuration and applying
the identical 64-position protocol and signal normalization.
