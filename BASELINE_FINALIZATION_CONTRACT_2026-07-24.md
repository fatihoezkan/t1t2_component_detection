# Baseline finalization contract — 2026-07-24

This document defines what “baseline complete” means for the single-voxel T1–T2 DETR. It is a
new artifact; the project’s existing plans, reports, READMEs, and notebooks remain unchanged.

## Fixed scientific scope

- Signed 64-point IR-MSE signals in the stored scanner order.
- Additive real-valued Gaussian noise.
- Tissue-agnostic random compartments satisfying `T1 > T2`.
- One, two, or three compartments per voxel, balanced across the training split.
- The existing ten-query DETR and Hungarian training loss.
- No signal-consistency loss, post-processing, query-count sweep, or loss-weight optimization in
  the baseline.

Those exclusions are deliberate. They keep the reference model interpretable so later changes can
be evaluated as controlled ablations.

## Completion criteria

The baseline is complete only when all of the following are produced from the selected checkpoint:

1. The exact training configuration, checkpoint identity, data manifests, split sizes, runtime
   versions, and random-seed provenance are recorded.
2. The existence threshold is selected on the validation split only. That threshold is frozen
   before test metrics are computed.
3. Test reporting includes:
   - compartment-count accuracy and MAE;
   - existence precision, recall, F1, false-positive count, and missed-compartment count;
   - matched T1, T2, and weight errors in physical units and relative units;
   - results separated by true compartment count;
   - physical-constraint violations;
   - signal-reconstruction error against the observed normalized signal.
4. The paired fixed-SNR ladder is evaluated with the same frozen threshold, and SNR 20 is marked as
   extrapolation because training starts at SNR 30.
5. The following figures are generated without manual notebook steps:
   - training and validation curves;
   - validation-threshold calibration;
   - test count confusion;
   - error by true compartment count;
   - SNR robustness;
   - predicted-versus-true T1/T2;
   - query activity and matched-query specialization;
   - deterministic representative successes and failures.
6. Automated tests cover threshold calibration, count/existence metrics, signal reconstruction,
   and the offline finalization smoke path.

## Data-selection rule

The cluster baseline used the first 33,333 train, 3,333 validation, 3,333 test, and 1,667
fixed-SNR voxels per compartment count with base seed 0. The locally available `full_1to4`
families use the same counter-based random streams, split codes, protocol, and base seed. Their
corresponding prefixes are therefore the exact baseline voxels, not a replacement sample. The
finalization configuration records the prefix lengths explicitly and validates the manifests
before evaluation.

## Decision boundary after completion

Only after this contract passes should the project start controlled experiments for:

1. query counts 5/10/15/20;
2. a limited Optuna search using validation data only;
3. physics-consistency loss;
4. post-processing as a separately reported comparison;
5. additional noise and compartment-separation stress tests.

The test split must not select any of those configurations.
