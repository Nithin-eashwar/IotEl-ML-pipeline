# Leakage-Resistant NILM Evaluation Design

## Goal

Measure how well the NILM classifiers generalize within the recordings currently
available, while reducing optimistic scores caused by overlapping windows,
chronological similarity, and recording-specific temperature or vibration
features.

This evaluation cannot prove cross-session generalization because each class has
only one recording. A future independent-recording test remains necessary.

## Data Split

Each class recording is split chronologically at the raw-row level:

- The first 80% is the development region.
- The final 20% is the untouched test region.
- A gap of at least one full window (128 samples) separates the regions.
- Windows are created only after splitting, so no raw sample can occur in both
  development and test windows.

Model selection uses blocked chronological folds inside each class's development
region. Validation blocks are contiguous, and one full-window gap is excluded
on both sides of each validation block. This prevents overlapping or immediately
adjacent windows from crossing a training/validation boundary.

The test region is evaluated only after a model and feature set have been
selected from development validation.

## Feature Ablations

Evaluate the following variants using identical splits:

1. All engineered features.
2. All features except temperature means (`ntc_mean`, `tmp117_mean`).
3. All features except accelerometer FFT bins (`accel_*_fft_*`).
4. All features except temperature means and accelerometer FFT bins.
5. Current and power features only (`cur_*`, `pwr_*`).

The comparison reports both aggregate and per-class behavior. A substantial
drop after removing a feature family indicates that the family carries much of
the predictive signal, but does not by itself prove that the signal is
load-driven.

## Models

Compare the existing KNN pipeline with a mildly regularized Random Forest:

- KNN retains standard scaling and distance weighting.
- Random Forest uses bounded depth and a larger minimum leaf size to reduce its
  ability to fit small recording-specific regions.

Classifier and feature-variant selection use development validation macro F1,
with accuracy shown as a secondary metric. Macro F1 gives each load class equal
importance. If validation macro F1 and accuracy tie, prefer the variant with
fewer features.

## Reporting

For every feature variant and classifier, print:

- Mean blocked-validation accuracy.
- Mean blocked-validation macro F1.
- Per-fold scores.

For the best development configuration, print untouched-test:

- Accuracy.
- Macro F1.
- Per-class precision, recall, and F1.
- Confusion matrix.

Also print a compact test ablation table by fitting each variant's best
classifier on the complete development region and evaluating it once on the
same untouched test region. This table is diagnostic rather than an additional
model-selection source.

## Safeguards

- Never shuffle windows across chronological boundaries.
- Never form windows before raw-row splitting.
- Never use test performance to choose the winning model.
- Do not print an in-sample classification report as evidence of
  generalization.
- Fail with a clear message if a class has too few rows or windows for the
  requested split.

## Limitations

The test data still comes from the same recording as the training data. The
result estimates later-in-recording performance, not performance on a new
mounting, motor, sensor, day, or operating environment. Multiple independent
recordings per class and grouped validation are required for that stronger
claim.
