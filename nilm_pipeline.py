"""
NILM Classifier Pipeline
========================
Cleans raw CSV data, engineers windowed features, and compares Random Forest
and KNN classifiers with leakage-resistant chronological evaluation.

Changes vs original:
  - Class 2 trimmed to largest contiguous segment (rows 2710+) to remove
    the concatenated sub-recordings identified by timestamp reset.
  - Cross-session test added for Class 3 (heavy_load) using a separate
    recording in class3_heavy_load_test.csv.

Run: python nilm_pipeline.py
"""

import os
import warnings

# Avoid unreliable physical-core detection in some restricted Windows shells.
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

import numpy as np
import pandas as pd
from scipy.fft import rfft, rfftfreq
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from sklearn.model_selection import cross_validate
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")


# ─── CONFIG ───────────────────────────────────────────────────────────────────
FILE_MAP = {
    1: "class1_unloaded__1_.csv",
    2: "class2_light_load__1_.csv",
    3: "class3_heavy_load__1_.csv",
    4: "class4_stall.csv",
}

# Cross-session test files: {class_label: (filepath, encoding)}
# Only class 3 has a separate-session recording right now.
CROSS_SESSION_MAP = {
    3: ("class3_heavy_load_test.csv", "utf-16"),
}

CLASS_NAMES = {
    1: "unloaded",
    2: "light_load",
    3: "heavy_load",
    4: "stall",
}

COLS = [
    "timestamp_ms",
    "accel_x",
    "accel_y",
    "accel_z",
    "accel_rms",
    "tmp117_c",
    "ntc_c",
    "ina_a_ma",
    "ina_a_mw",
    "ina_b_ma",
    "ina_b_mw",
    "motor_rpm",
    "load_class",
]

# 128 samples at 100 Hz = 1.28 seconds per window.
WINDOW_SIZE = 128
STEP_SIZE = 64
SAMPLE_RATE = 100

DEAD_CURRENT_THRESHOLD = 2.0
DEVELOPMENT_FRACTION = 0.80
N_CV_SPLITS = 5
PURGE_GAP = WINDOW_SIZE

# Minimum gap (ms) that marks a hard boundary between sub-recordings.
SEGMENT_GAP_MS = 500


# ─── DATA LOADING ─────────────────────────────────────────────────────────────
def _parse_and_filter(df, label):
    """Numeric coerce, NaN drop, dead-current filter, column drop."""
    for column in COLS:
        df[column] = pd.to_numeric(df[column], errors="coerce")
    df = df.dropna(subset=["timestamp_ms", "ina_a_ma"]).reset_index(drop=True)
    df = df[df["ina_a_ma"].abs() >= DEAD_CURRENT_THRESHOLD].copy()
    df = df.drop(columns=["ina_b_ma", "ina_b_mw", "motor_rpm"])
    df["load_class"] = label
    return df.reset_index(drop=True)


def _largest_contiguous_segment(df):
    """
    Return the largest contiguous block of rows with no timestamp gap
    larger than SEGMENT_GAP_MS between consecutive samples.
    """
    ts = df["timestamp_ms"].values
    diffs = np.diff(ts)
    gap_indices = np.where(diffs > SEGMENT_GAP_MS)[0]

    if len(gap_indices) == 0:
        return df  # already one contiguous block

    boundaries = [0] + list(gap_indices + 1) + [len(df)]
    segments = [
        (boundaries[i], boundaries[i + 1])
        for i in range(len(boundaries) - 1)
    ]
    start, end = max(segments, key=lambda se: se[1] - se[0])
    return df.iloc[start:end].reset_index(drop=True)


def load_and_clean(filepath, label, encoding="utf-8", keep_largest_segment=False):
    df = pd.read_csv(
        filepath, header=None, names=COLS, on_bad_lines="skip", encoding=encoding
    )
    before_filter = len(df)
    df = _parse_and_filter(df, label)

    if keep_largest_segment:
        df_full = df.copy()
        df = _largest_contiguous_segment(df)
        seg_note = (
            f", trimmed to largest contiguous segment "
            f"({len(df_full) - len(df)} rows removed)"
        )
    else:
        seg_note = ""

    after = len(df)
    dropped = before_filter - after
    dropped_pct = 100 * dropped / before_filter if before_filter else 0.0
    print(
        f"  Class {label}: {before_filter} rows -> {after} clean "
        f"({dropped} dead-current rows dropped, {dropped_pct:.1f}%{seg_note})"
    )
    return df.reset_index(drop=True)


# ─── FEATURE ENGINEERING ──────────────────────────────────────────────────────
def compute_fft_features(signal, sample_rate, n_bins=8):
    """Return spectral energy in equally spaced frequency bands."""
    n = len(signal)
    if n < 4:
        return [0.0] * n_bins

    signal = signal - np.mean(signal)
    magnitudes = np.abs(rfft(signal))
    freqs = rfftfreq(n, d=1.0 / sample_rate)

    nyquist = sample_rate / 2.0
    bin_edges = np.linspace(0, nyquist, n_bins + 1)
    bin_features = []
    for i in range(n_bins):
        mask = (freqs >= bin_edges[i]) & (freqs < bin_edges[i + 1])
        bin_features.append(float(np.sum(magnitudes[mask] ** 2)))
    return bin_features


def extract_window_features(window_df):
    feats = {}

    cur = window_df["ina_a_ma"].values
    feats["cur_mean"] = np.mean(cur)
    feats["cur_std"] = np.std(cur)
    feats["cur_min"] = np.min(cur)
    feats["cur_max"] = np.max(cur)
    feats["cur_range"] = np.max(cur) - np.min(cur)
    feats["cur_rms"] = np.sqrt(np.mean(cur**2))
    feats["cur_p25"] = np.percentile(cur, 25)
    feats["cur_p75"] = np.percentile(cur, 75)
    feats["cur_iqr"] = feats["cur_p75"] - feats["cur_p25"]

    pwr = window_df["ina_a_mw"].values
    feats["pwr_mean"] = np.mean(pwr)
    feats["pwr_std"] = np.std(pwr)
    feats["pwr_range"] = np.max(pwr) - np.min(pwr)

    for axis in ["accel_x", "accel_y", "accel_z"]:
        sig = window_df[axis].values
        feats[f"{axis}_std"] = np.std(sig)
        feats[f"{axis}_range"] = np.max(sig) - np.min(sig)
        for i, value in enumerate(
            compute_fft_features(sig, SAMPLE_RATE, n_bins=8)
        ):
            feats[f"{axis}_fft_{i}"] = value

    rms = window_df["accel_rms"].values
    feats["accel_rms_mean"] = np.mean(rms)
    feats["accel_rms_std"] = np.std(rms)

    feats["ntc_mean"] = np.mean(window_df["ntc_c"].values)
    feats["tmp117_mean"] = np.mean(window_df["tmp117_c"].values)
    return feats


def build_feature_matrix(df):
    """Create windows and retain their raw-row intervals for purged CV."""
    feature_rows = []
    labels = []
    starts = []

    for start in range(0, len(df) - WINDOW_SIZE + 1, STEP_SIZE):
        end = start + WINDOW_SIZE
        window = df.iloc[start:end]
        feature_rows.append(extract_window_features(window))
        labels.append(int(window["load_class"].mode()[0]))
        starts.append(start)

    return (
        pd.DataFrame(feature_rows),
        np.asarray(labels, dtype=int),
        np.asarray(starts, dtype=int),
    )


# ─── SPLITTING ────────────────────────────────────────────────────────────────
def split_recordings(clean_data):
    """Split each class chronologically before creating any windows."""
    development = {}
    test = {}

    print("\n[2] Chronological raw-row split (within-session)...")
    for label, df in clean_data.items():
        test_start = int(len(df) * DEVELOPMENT_FRACTION)
        development_end = test_start - PURGE_GAP

        if development_end < WINDOW_SIZE or len(df) - test_start < WINDOW_SIZE:
            raise ValueError(
                f"Class {label} does not contain enough rows for the "
                "development/test split and purge gap."
            )

        development[label] = df.iloc[:development_end].reset_index(drop=True)
        test[label] = df.iloc[test_start:].reset_index(drop=True)
        print(
            f"  Class {label}: development={development_end} rows, "
            f"gap={PURGE_GAP} rows, test={len(df) - test_start} rows"
        )

    return development, test


def build_dataset(data_by_class):
    feature_frames = []
    label_arrays = []
    metadata_frames = []

    for label, df in data_by_class.items():
        X_class, y_class, starts = build_feature_matrix(df)
        if X_class.empty:
            raise ValueError(f"Class {label} produced no complete windows.")

        feature_frames.append(X_class)
        label_arrays.append(y_class)
        metadata_frames.append(
            pd.DataFrame(
                {
                    "class_label": label,
                    "row_start": starts,
                    "row_end": starts + WINDOW_SIZE,
                }
            )
        )

    X = pd.concat(feature_frames, ignore_index=True).fillna(0)
    y = np.concatenate(label_arrays)
    metadata = pd.concat(metadata_frames, ignore_index=True)
    return X, y, metadata


# ─── CROSS-VALIDATION SPLITS ──────────────────────────────────────────────────
def make_purged_blocked_splits(metadata, development):
    """
    Create chronological validation blocks with a full-window purge on both
    sides. The same fold number represents the same time region in every class.
    """
    splits = []

    for fold in range(N_CV_SPLITS):
        train_mask = np.zeros(len(metadata), dtype=bool)
        validation_mask = np.zeros(len(metadata), dtype=bool)

        for label, df in development.items():
            row_edges = np.linspace(0, len(df), N_CV_SPLITS + 1, dtype=int)
            validation_start = row_edges[fold]
            validation_end = row_edges[fold + 1]

            class_mask = metadata["class_label"].to_numpy() == label
            starts = metadata["row_start"].to_numpy()
            ends = metadata["row_end"].to_numpy()

            validation_mask |= (
                class_mask
                & (starts >= validation_start)
                & (ends <= validation_end)
            )
            train_mask |= class_mask & (
                (ends <= validation_start - PURGE_GAP)
                | (starts >= validation_end + PURGE_GAP)
            )

        train_indices = np.flatnonzero(train_mask)
        validation_indices = np.flatnonzero(validation_mask)

        if not len(train_indices) or not len(validation_indices):
            raise ValueError(f"Fold {fold + 1} has an empty train or validation set.")

        train_classes = set(metadata.iloc[train_indices]["class_label"])
        validation_classes = set(metadata.iloc[validation_indices]["class_label"])
        expected_classes = set(development)
        if train_classes != expected_classes or validation_classes != expected_classes:
            raise ValueError(f"Fold {fold + 1} does not contain every class.")

        if np.any(train_mask & validation_mask):
            raise AssertionError(
                f"Fold {fold + 1} has windows assigned to both train and validation."
            )

        splits.append((train_indices, validation_indices))

    return splits


# ─── FEATURE VARIANTS ─────────────────────────────────────────────────────────
def get_feature_variants(columns):
    columns = list(columns)
    temperature = {"ntc_mean", "tmp117_mean"}
    accel_fft = {
        column
        for column in columns
        if column.startswith("accel_") and "_fft_" in column
    }

    return {
        "All features": columns,
        "No temperature": [
            column for column in columns if column not in temperature
        ],
        "No accel FFT": [
            column for column in columns if column not in accel_fft
        ],
        "No temp or accel FFT": [
            column
            for column in columns
            if column not in temperature and column not in accel_fft
        ],
        "Current + power only": [
            column
            for column in columns
            if column.startswith("cur_") or column.startswith("pwr_")
        ],
    }


# ─── MODELS ───────────────────────────────────────────────────────────────────
def get_models():
    return {
        "Random Forest": Pipeline(
            [
                (
                    "clf",
                    RandomForestClassifier(
                        n_estimators=200,
                        max_depth=12,
                        min_samples_leaf=5,
                        class_weight="balanced",
                        random_state=42,
                        n_jobs=1,
                    ),
                )
            ]
        ),
        "KNN": Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "clf",
                    KNeighborsClassifier(
                        n_neighbors=7,
                        weights="distance",
                        metric="euclidean",
                    ),
                ),
            ]
        ),
    }


# ─── EVALUATION ───────────────────────────────────────────────────────────────
def evaluate_development(X, y, splits, feature_variants):
    results = []
    scoring = {"accuracy": "accuracy", "macro_f1": "f1_macro"}

    print("\n[4] Purged blocked validation and feature ablation...")
    for variant_name, feature_columns in feature_variants.items():
        print(f"\n  {variant_name} ({len(feature_columns)} features)")
        for model_name, model in get_models().items():
            scores = cross_validate(
                model,
                X[feature_columns],
                y,
                cv=splits,
                scoring=scoring,
                n_jobs=None,
                error_score="raise",
            )
            accuracy = scores["test_accuracy"]
            macro_f1 = scores["test_macro_f1"]
            print(
                f"    {model_name:<14} "
                f"accuracy={accuracy.mean() * 100:5.1f}% "
                f"+/- {accuracy.std() * 100:4.1f}%  "
                f"macro-F1={macro_f1.mean() * 100:5.1f}% "
                f"+/- {macro_f1.std() * 100:4.1f}%"
            )
            print(
                " " * 18
                + "fold F1="
                + str([f"{score * 100:.1f}%" for score in macro_f1])
            )
            results.append(
                {
                    "variant": variant_name,
                    "features": feature_columns,
                    "model_name": model_name,
                    "model": model,
                    "cv_accuracy": accuracy.mean(),
                    "cv_macro_f1": macro_f1.mean(),
                }
            )

    return results


def choose_best_result(results):
    return max(
        results,
        key=lambda result: (
            result["cv_macro_f1"],
            result["cv_accuracy"],
            -len(result["features"]),
        ),
    )


def evaluate_test_ablations(results, X_dev, y_dev, X_test, y_test, section_label):
    """Evaluate each CV-selected feature variant once on a held-out test set."""
    best_by_variant = {}
    for result in results:
        current = best_by_variant.get(result["variant"])
        if current is None or (
            result["cv_macro_f1"],
            result["cv_accuracy"],
        ) > (
            current["cv_macro_f1"],
            current["cv_accuracy"],
        ):
            best_by_variant[result["variant"]] = result

    labels = sorted(np.unique(y_test))
    rows = []
    fitted = {}

    for variant_name, result in best_by_variant.items():
        model = clone(result["model"])
        columns = result["features"]
        model.fit(X_dev[columns], y_dev)
        predictions = model.predict(X_test[columns])
        per_class_f1 = precision_recall_fscore_support(
            y_test,
            predictions,
            labels=labels,
            zero_division=0,
        )[2]

        rows.append(
            {
                "variant": variant_name,
                "model": result["model_name"],
                "accuracy": accuracy_score(y_test, predictions),
                "macro_f1": f1_score(
                    y_test,
                    predictions,
                    average="macro",
                    zero_division=0,
                ),
                "per_class_f1": dict(zip(labels, per_class_f1)),
            }
        )
        fitted[variant_name] = (model, predictions)

    print(f"\n{section_label}")
    class_headers = "  ".join(
        f"{CLASS_NAMES[label]} F1" for label in labels
    )
    print(
        f"  {'Feature variant':<24} {'Model':<14} "
        f"{'Acc':>7} {'Macro-F1':>9}  {class_headers}"
    )
    for row in rows:
        class_values = "  ".join(
            f"{row['per_class_f1'].get(label, float('nan')) * 100:10.1f}%"
            for label in labels
        )
        print(
            f"  {row['variant']:<24} {row['model']:<14} "
            f"{row['accuracy'] * 100:6.1f}% "
            f"{row['macro_f1'] * 100:8.1f}%  {class_values}"
        )

    return rows, fitted


def print_confusion_matrix(y_true, y_pred, labels):
    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    header = " " * 14 + "  ".join(
        f"{CLASS_NAMES[label]:>10}" for label in labels
    )
    print(header)
    for label, row in zip(labels, matrix):
        values = "  ".join(f"{value:>10}" for value in row)
        print(f"  {CLASS_NAMES[label]:>10}  {values}")


def print_feature_importances(model, features, n=15):
    rf_model = model.named_steps["clf"]
    importances = pd.Series(rf_model.feature_importances_, index=features)
    for feature, importance in (
        importances.sort_values(ascending=False).head(n).items()
    ):
        print(f"  {feature:>25}  {importance:.4f}")


# ─── CROSS-SESSION TEST ───────────────────────────────────────────────────────
def load_cross_session_data():
    """
    Load the cross-session test files and return a dict of
    {label: DataFrame} using the same cleaning pipeline.
    """
    cross_data = {}
    print("\n[A] Loading cross-session test data...")
    for label, (filepath, encoding) in CROSS_SESSION_MAP.items():
        df = pd.read_csv(
            filepath, header=None, names=COLS, on_bad_lines="skip", encoding=encoding
        )
        before = len(df)
        df = _parse_and_filter(df, label)
        after = len(df)
        ts = df["timestamp_ms"].values
        n_windows = max(0, (len(df) - WINDOW_SIZE) // STEP_SIZE + 1)
        print(
            f"  Class {label} ({CLASS_NAMES[label]}): {before} raw -> {after} clean rows  "
            f"ts {ts[0]:.0f}–{ts[-1]:.0f} ms ({(ts[-1]-ts[0])/1000:.1f}s)  "
            f"~{n_windows} windows"
        )
        cross_data[label] = df
    return cross_data


def evaluate_cross_session(best, X_dev, y_dev, cross_data):
    """
    Train the best model on the full development set, then predict on
    the cross-session windows for the available classes only.
    Prints detailed numbers: accuracy per window, per-sample raw stats,
    and a confusion matrix (for the subset of labels available).
    """
    columns = best["features"]
    model = clone(best["model"])
    model.fit(X_dev[columns], y_dev)

    print("\n[B] Cross-session test — per-class window predictions")
    print(f"  Model: {best['model_name']}  |  Features: {best['variant']}")
    print(f"  {'Class':<12} {'Windows':>7}  {'Correct':>7}  {'Acc':>7}  "
          f"{'Pred distribution'}")

    all_true = []
    all_pred = []

    for label, df in cross_data.items():
        X_cs, y_cs, _ = build_feature_matrix(df)
        if X_cs.empty:
            print(f"  Class {label}: no complete windows (need >= {WINDOW_SIZE} rows)")
            continue

        # Align columns (fill any missing with 0)
        X_cs = X_cs.reindex(columns=columns, fill_value=0)
        preds = model.predict(X_cs)
        correct = np.sum(preds == label)
        acc = correct / len(preds) * 100

        # Distribution of predictions
        pred_counts = {
            CLASS_NAMES[l]: np.sum(preds == l)
            for l in sorted(CLASS_NAMES)
            if np.sum(preds == l) > 0
        }
        dist_str = "  ".join(f"{k}={v}" for k, v in pred_counts.items())

        print(
            f"  {CLASS_NAMES[label]:<12} {len(preds):>7}  {correct:>7}  "
            f"{acc:>6.1f}%  {dist_str}"
        )
        all_true.extend([label] * len(preds))
        all_pred.extend(preds)

    if not all_true:
        print("  No windows available for cross-session evaluation.")
        return

    all_true = np.array(all_true)
    all_pred = np.array(all_pred)
    overall_acc = accuracy_score(all_true, all_pred) * 100
    overall_f1 = f1_score(all_true, all_pred, average="macro", zero_division=0) * 100
    print(f"\n  Overall cross-session  Acc={overall_acc:.1f}%  macro-F1={overall_f1:.1f}%")

    print("\n[C] Cross-session detailed report (for available classes)...")
    labels_present = sorted(np.unique(all_true))
    target_names = [CLASS_NAMES[l] for l in labels_present]
    print(
        classification_report(
            all_true, all_pred,
            labels=labels_present,
            target_names=target_names,
            zero_division=0,
        )
    )

    print("  Confusion Matrix (cross-session):")
    print_confusion_matrix(all_true, all_pred, labels_present)

    print("\n[D] Raw signal statistics — session A vs cross-session (class 3 only)")
    print(f"  {'Metric':<20} {'Session A':>12}  {'Cross-session':>14}")
    print("  " + "-" * 50)

    # Pull session A class-3 data for comparison
    sa_df = pd.read_csv(FILE_MAP[3], header=None, names=COLS, on_bad_lines="skip")
    for c in COLS:
        sa_df[c] = pd.to_numeric(sa_df[c], errors="coerce")
    sa_df = sa_df.dropna(subset=["timestamp_ms", "ina_a_ma"])
    sa_df = sa_df[sa_df["ina_a_ma"].abs() >= DEAD_CURRENT_THRESHOLD]

    cs_df = cross_data.get(3)
    if cs_df is not None:
        for col, label_str in [
            ("ina_a_ma", "Current mean (mA)"),
            ("ina_a_mw", "Power mean (mW)"),
            ("accel_rms", "Accel RMS mean"),
        ]:
            sa_val = sa_df[col].mean() if col in sa_df.columns else float("nan")
            cs_val = cs_df[col].mean() if col in cs_df.columns else float("nan")
            print(f"  {label_str:<20} {sa_val:>12.3f}  {cs_val:>14.3f}")

        for col, label_str in [
            ("ina_a_ma", "Current std (mA)"),
            ("accel_rms", "Accel RMS std"),
        ]:
            sa_val = sa_df[col].std() if col in sa_df.columns else float("nan")
            cs_val = cs_df[col].std() if col in cs_df.columns else float("nan")
            print(f"  {label_str:<20} {sa_val:>12.3f}  {cs_val:>14.3f}")

        # Window-level accel_rms_std comparison
        def window_accel_rms_std(df_):
            stds = []
            rms = df_["accel_rms"].values
            for s in range(0, len(rms) - WINDOW_SIZE + 1, STEP_SIZE):
                stds.append(np.std(rms[s:s + WINDOW_SIZE]))
            return np.array(stds)

        sa_ws = window_accel_rms_std(sa_df)
        cs_ws = window_accel_rms_std(cs_df)
        print(
            f"  {'accel_rms_std/win':<20} {sa_ws.mean():>12.4f}  {cs_ws.mean():>14.4f}"
        )


# ─── MAIN ─────────────────────────────────────────────────────────────────────
def main():
    print("=" * 72)
    print("  NILM Pipeline - Leakage-Resistant Load Classification")
    print("=" * 72)
    print("\n  [WARNING] within-session test scores reflect a single recording.")
    print("  [WARNING] Cross-session section [B-D] gives the real generalisation picture.")

    # ── Step 1: load & clean training data ───────────────────────────────────
    print("\n[1] Loading and cleaning data...")
    clean_data = {}
    for label, filename in FILE_MAP.items():
        # Class 2 has a timestamp reset — keep only the largest contiguous block.
        keep_largest = label == 2
        clean_data[label] = load_and_clean(
            filename, label, keep_largest_segment=keep_largest
        )

    print(f"\n  Total clean rows: {sum(map(len, clean_data.values()))}")
    for label, df in clean_data.items():
        print(
            f"    Class {label} ({CLASS_NAMES[label]}): "
            f"{len(df)} rows (~{len(df) / SAMPLE_RATE:.0f}s)"
        )

    # ── Step 2-3: split & feature extraction ─────────────────────────────────
    development, test = split_recordings(clean_data)

    print("\n[3] Extracting windows after the split...")
    print(
        f"  Window={WINDOW_SIZE} samples ({WINDOW_SIZE / SAMPLE_RATE:.2f}s), "
        f"step={STEP_SIZE}, overlap=50%, purge gap={PURGE_GAP} samples"
    )
    X_dev, y_dev, dev_metadata = build_dataset(development)
    X_test, y_test, _ = build_dataset(test)
    print(f"  Development matrix: {X_dev.shape}")
    print(f"  Test matrix:        {X_test.shape}")
    for label in FILE_MAP:
        print(
            f"    Class {label}: "
            f"{np.sum(y_dev == label)} development windows, "
            f"{np.sum(y_test == label)} test windows"
        )

    splits = make_purged_blocked_splits(dev_metadata, development)
    for fold_number, (train_indices, validation_indices) in enumerate(
        splits, start=1
    ):
        print(
            f"    Fold {fold_number}: {len(train_indices)} train windows, "
            f"{len(validation_indices)} validation windows"
        )

    # ── Step 4: development CV ───────────────────────────────────────────────
    feature_variants = get_feature_variants(X_dev.columns)
    results = evaluate_development(X_dev, y_dev, splits, feature_variants)
    best = choose_best_result(results)

    print(
        "\n  Development winner: "
        f"{best['model_name']} with {best['variant']} "
        f"(macro-F1={best['cv_macro_f1'] * 100:.1f}%, "
        f"accuracy={best['cv_accuracy'] * 100:.1f}%)"
    )

    # ── Step 5: within-session test ──────────────────────────────────────────
    _, fitted = evaluate_test_ablations(
        results,
        X_dev,
        y_dev,
        X_test,
        y_test,
        section_label="[5] Untouched chronological (within-session) test ablation...",
    )

    best_model, best_predictions = fitted[best["variant"]]
    labels = sorted(np.unique(y_test))
    target_names = [CLASS_NAMES[label] for label in labels]

    print("\n[6] Detailed untouched-test report for development winner...")
    print(
        classification_report(
            y_test,
            best_predictions,
            labels=labels,
            target_names=target_names,
            zero_division=0,
        )
    )
    print("  Confusion Matrix:")
    print_confusion_matrix(y_test, best_predictions, labels)

    if best["model_name"] == "Random Forest":
        print("\n[7] Top 15 Random Forest features...")
        print_feature_importances(best_model, best["features"])

    # ── Cross-session evaluation ──────────────────────────────────────────────
    cross_data = load_cross_session_data()
    evaluate_cross_session(best, X_dev, y_dev, cross_data)

    print("\n" + "=" * 72)
    print("  Pipeline complete.")
    print(
        "  Note: within-session test = later data from same recordings.\n"
        "        Cross-session test  = separate recording (different session)."
    )
    print("=" * 72)


if __name__ == "__main__":
    main()
