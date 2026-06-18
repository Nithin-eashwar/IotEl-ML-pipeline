"""
NILM Classifier Pipeline
========================
Cleans raw CSV data, engineers windowed features, and compares Random Forest
and KNN classifiers with leakage-resistant chronological evaluation.

Changes vs original:
  - Class 2 trimmed to largest contiguous segment to remove timestamp-reset artefact.
  - Cross-session test using new-session CSVs for classes 2, 3, 4 (batched).
  - Matplotlib confusion matrix saved to disk for easy sharing.

Run: python nilm_pipeline.py
"""

import argparse
import os
import textwrap
import warnings

# Avoid unreliable physical-core detection in some restricted Windows shells.
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

import matplotlib
matplotlib.use("Agg")  # non-interactive backend — safe for all environments
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
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
# New-session recordings for classes 1, 2, 3, 4.
CROSS_SESSION_MAP = {
    1: ("1_unloaded_test.csv",     "utf-8"),
    2: ("2_light_load_test.csv",   "utf-8"),
    3: ("3_heavy_load_test.csv",   "utf-8"),
    4: ("4_stall_test.csv",        "utf-8"),
}

# SESSION_MAP: both recording sessions used for LOSO.
# Format: {session_id: {class_label: (filepath, encoding, keep_largest_segment)}}
# keep_largest_segment=True trims to the longest contiguous block (handles
# timestamp resets and files with extra trailing commas/gaps).
SESSION_MAP = {
    "A": {
        1: ("class1_unloaded__1_.csv",   "utf-8", False),
        2: ("class2_light_load__1_.csv", "utf-8", True),   # has internal timestamp reset
        3: ("class3_heavy_load__1_.csv", "utf-8", False),
        4: ("class4_stall.csv",          "utf-8", False),
    },
    "B": {
        1: ("1_unloaded_test.csv",       "utf-8", False),  # no real gaps; segment trim causes fragmentation
        2: ("2_light_load_test.csv",     "utf-8", True),
        3: ("3_heavy_load_test.csv",     "utf-8", True),
        4: ("4_stall_test.csv",          "utf-8", True),
    },
}

# Where to save output figures and find data (robust to script location).
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "..", "data")
OUTPUT_DIR = os.path.join(BASE_DIR, "..", "results")

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
    full_path = os.path.join(DATA_DIR, filepath)
    df = pd.read_csv(
        full_path, header=None, names=COLS, on_bad_lines="skip", encoding=encoding
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


def load_session_file(filepath, label, encoding="utf-8", keep_largest=False):
    """
    Unified loader for both Session A and Session B files.
    Some files start with 9 columns, some with 13, 17, or 19.
    By providing names=range(25), pandas will read all rows regardless of length
    (padding short rows with NaN and accepting up to 25 columns).
    We then slice the first 13 columns to match our expected schema.
    """
    full_path = os.path.join(DATA_DIR, filepath)
    raw = pd.read_csv(
        full_path, header=None, names=range(25), on_bad_lines="skip", encoding=encoding
    )
    raw = raw.iloc[:, : len(COLS)]  # keep only the first 13 columns
    raw.columns = COLS
    before = len(raw)
    df = _parse_and_filter(raw, label)
    if keep_largest:
        df = _largest_contiguous_segment(df)
    print(
        f"    Class {label} ({CLASS_NAMES[label]}): "
        f"{before} raw -> {len(df)} clean rows"
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
            print(
                f"  [WARN] Class {label} ({CLASS_NAMES.get(label, label)}): "
                f"fewer than {WINDOW_SIZE} rows after cleaning — skipped."
            )
            continue

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
def evaluate_development(X, y, splits, feature_variants,
                         header="[4] Purged blocked validation and feature ablation..."):
    results = []
    scoring = {"accuracy": "accuracy", "macro_f1": "f1_macro"}

    print(f"\n{header}")
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


def choose_best_random_forest_result(results):
    rf_results = [
        result for result in results if result["model_name"] == "Random Forest"
    ]
    if not rf_results:
        raise ValueError("No Random Forest result is available for firmware export.")
    return choose_best_result(rf_results)


def _c_string(value):
    return '"' + value.replace("\\", "\\\\").replace('"', '\\"') + '"'


def _format_float(value):
    if not np.isfinite(value):
        return "0.0f"
    s = f"{float(value):.9g}"
    if "." not in s:
        s += ".0"
    return s + "f"


def export_random_forest_firmware(model, feature_cols, class_names, output_dir):
    """
    Export a fitted sklearn RandomForestClassifier as plain C arrays.

    The firmware walks each tree until a leaf, sums per-tree class
    probabilities, then returns the class with the largest average vote.
    """
    clf = model.named_steps.get("clf")
    if not isinstance(clf, RandomForestClassifier):
        raise TypeError("Firmware export requires a fitted RandomForestClassifier.")

    os.makedirs(output_dir, exist_ok=True)
    header_path = os.path.join(output_dir, "nilm_model.h")
    source_path = os.path.join(output_dir, "nilm_model.c")
    classes = [int(value) for value in clf.classes_]
    class_count = len(classes)

    header = f"""\
    #pragma once

    #include <stddef.h>

    #ifdef __cplusplus
    extern "C" {{
    #endif

    #define NILM_MODEL_FEATURE_COUNT {len(feature_cols)}
    #define NILM_MODEL_CLASS_COUNT {class_count}

    size_t nilm_model_feature_count(void);
    const char *nilm_model_feature_name(size_t index);
    int nilm_model_predict(const float features[NILM_MODEL_FEATURE_COUNT],
                           float *confidence);
    const char *nilm_model_class_name(int class_id);

    #ifdef __cplusplus
    }}
    #endif
    """

    source_lines = [
        '#include "nilm_model.h"',
        "",
        "typedef struct {",
        "    int feature;",
        "    float threshold;",
        "    int left;",
        "    int right;",
        f"    float proba[{class_count}];",
        "} NilmTreeNode;",
        "",
        "typedef struct {",
        "    const NilmTreeNode *nodes;",
        "    int node_count;",
        "} NilmTree;",
        "",
        "static const char *const FEATURE_NAMES[NILM_MODEL_FEATURE_COUNT] = {",
    ]
    source_lines.extend(f"    {_c_string(name)}," for name in feature_cols)
    source_lines.extend(
        [
            "};",
            "",
            f"static const int CLASS_IDS[NILM_MODEL_CLASS_COUNT] = "
            f"{{{', '.join(map(str, classes))}}};",
            "",
            "static const char *const CLASS_NAMES[NILM_MODEL_CLASS_COUNT] = {",
        ]
    )
    source_lines.extend(
        f"    {_c_string(class_names.get(class_id, str(class_id)))},"
        for class_id in classes
    )
    source_lines.append("};")

    for tree_index, estimator in enumerate(clf.estimators_):
        tree = estimator.tree_
        source_lines.extend(
            [
                "",
                f"static const NilmTreeNode TREE_{tree_index}_NODES[] = {{",
            ]
        )
        for node_index in range(tree.node_count):
            feature = int(tree.feature[node_index])
            threshold = _format_float(tree.threshold[node_index])
            left = int(tree.children_left[node_index])
            right = int(tree.children_right[node_index])
            values = tree.value[node_index][0]
            total = float(np.sum(values))
            if total > 0.0:
                probabilities = [value / total for value in values]
            else:
                probabilities = [0.0] * class_count
            proba = ", ".join(_format_float(value) for value in probabilities)
            source_lines.append(
                f"    {{{feature}, {threshold}, {left}, {right}, {{{proba}}}}},"
            )
        source_lines.append("};")

    source_lines.extend(["", "static const NilmTree TREES[] = {"])
    for tree_index, estimator in enumerate(clf.estimators_):
        source_lines.append(
            f"    {{TREE_{tree_index}_NODES, {estimator.tree_.node_count}}},"
        )
    source_lines.extend(
        [
            "};",
            "",
            "size_t nilm_model_feature_count(void)",
            "{",
            "    return NILM_MODEL_FEATURE_COUNT;",
            "}",
            "",
            "const char *nilm_model_feature_name(size_t index)",
            "{",
            "    return index < NILM_MODEL_FEATURE_COUNT ? FEATURE_NAMES[index] : \"\";",
            "}",
            "",
            "const char *nilm_model_class_name(int class_id)",
            "{",
            "    for (size_t i = 0; i < NILM_MODEL_CLASS_COUNT; ++i) {",
            "        if (CLASS_IDS[i] == class_id) return CLASS_NAMES[i];",
            "    }",
            "    return \"unknown\";",
            "}",
            "",
            "int nilm_model_predict(const float features[NILM_MODEL_FEATURE_COUNT],",
            "                       float *confidence)",
            "{",
            "    float votes[NILM_MODEL_CLASS_COUNT] = {0};",
            "    const int tree_count = (int)(sizeof(TREES) / sizeof(TREES[0]));",
            "",
            "    for (int tree_index = 0; tree_index < tree_count; ++tree_index) {",
            "        const NilmTree *tree = &TREES[tree_index];",
            "        int node_index = 0;",
            "",
            "        while (node_index >= 0 && node_index < tree->node_count) {",
            "            const NilmTreeNode *node = &tree->nodes[node_index];",
            "            if (node->feature < 0) {",
            "                for (size_t c = 0; c < NILM_MODEL_CLASS_COUNT; ++c) {",
            "                    votes[c] += node->proba[c];",
            "                }",
            "                break;",
            "            }",
            "            node_index = features[node->feature] <= node->threshold",
            "                ? node->left",
            "                : node->right;",
            "        }",
            "    }",
            "",
            "    size_t best = 0;",
            "    for (size_t c = 1; c < NILM_MODEL_CLASS_COUNT; ++c) {",
            "        if (votes[c] > votes[best]) best = c;",
            "    }",
            "    if (confidence != NULL) {",
            "        *confidence = tree_count > 0 ? votes[best] / (float)tree_count : 0.0f;",
            "    }",
            "    return CLASS_IDS[best];",
            "}",
            "",
        ]
    )

    with open(header_path, "w", encoding="utf-8") as f:
        f.write(textwrap.dedent(header))
    with open(source_path, "w", encoding="utf-8") as f:
        f.write("\n".join(source_lines))

    return header_path, source_path


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
    Handles CSVs that have extra trailing comma-separated columns.
    """
    cross_data = {}
    print("\n[A] Loading cross-session test data...")
    for label, (filepath, encoding) in CROSS_SESSION_MAP.items():
        # Read without fixed column names so extra trailing columns are tolerated
        full_path = os.path.join(DATA_DIR, filepath)
        raw = pd.read_csv(
            full_path, header=None, on_bad_lines="skip", encoding=encoding
        )
        # Keep only the first 13 columns (our expected schema)
        raw = raw.iloc[:, : len(COLS)]
        raw.columns = COLS
        before = len(raw)
        df = _parse_and_filter(raw, label)
        # If there is a timestamp gap, keep the largest contiguous segment
        df = _largest_contiguous_segment(df)
        after = len(df)
        ts = df["timestamp_ms"].values
        n_windows = max(0, (len(df) - WINDOW_SIZE) // STEP_SIZE + 1)
        print(
            f"  Class {label} ({CLASS_NAMES[label]}): {before} raw -> {after} clean rows  "
            f"({(ts[-1]-ts[0])/1000:.1f}s)  ~{n_windows} windows"
        )
        cross_data[label] = df
    return cross_data


def _save_confusion_matrix_figure(
    cm, labels, title, filepath, acc, macro_f1, model_name, feature_variant
):
    """Render and save a polished confusion matrix figure using matplotlib."""
    n = len(labels)
    fig, ax = plt.subplots(figsize=(max(6, n * 1.6), max(5, n * 1.4)))
    fig.patch.set_facecolor("#0f1117")
    ax.set_facecolor("#0f1117")

    # Normalised values for colour, raw counts for text
    row_sums = cm.sum(axis=1, keepdims=True)
    cm_norm = np.where(row_sums == 0, 0, cm / row_sums.astype(float))

    im = ax.imshow(cm_norm, interpolation="nearest", cmap="Blues", vmin=0, vmax=1)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))
    cbar.ax.tick_params(colors="#cccccc", labelsize=9)
    cbar.outline.set_edgecolor("#333333")

    # Cell annotations
    thresh = cm_norm.max() / 2.0
    for i in range(n):
        for j in range(n):
            pct = cm_norm[i, j] * 100
            count = cm[i, j]
            color = "white" if cm_norm[i, j] > thresh else "#cccccc"
            ax.text(
                j, i,
                f"{pct:.1f}%\n({count})",
                ha="center", va="center",
                fontsize=10, fontweight="bold", color=color,
            )

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=11, color="#eeeeee")
    ax.set_yticklabels(labels, fontsize=11, color="#eeeeee")
    ax.tick_params(axis="both", colors="#555555")
    for spine in ax.spines.values():
        spine.set_edgecolor("#333333")

    ax.set_xlabel("Predicted label", fontsize=12, color="#aaaaaa", labelpad=10)
    ax.set_ylabel("True label", fontsize=12, color="#aaaaaa", labelpad=10)

    subtitle = (
        f"Model: {model_name}  |  Features: {feature_variant}\n"
        f"Accuracy: {acc:.1f}%   Macro-F1: {macro_f1:.1f}%   "
        f"(classes tested: {', '.join(labels)})"
    )
    ax.set_title(
        f"{title}\n",
        fontsize=14, fontweight="bold", color="white", pad=12
    )
    fig.text(
        0.5, 0.94, subtitle,
        ha="center", va="center",
        fontsize=9, color="#aaaaaa",
    )

    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(filepath, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  Figure saved -> {filepath}")


def evaluate_cross_session(best, X_dev, y_dev, cross_data, output_dir):
    """
    Train the best model on the full development set, then predict on the
    cross-session windows for all available classes in a single batch.
    Saves a matplotlib confusion matrix figure.
    """
    os.makedirs(output_dir, exist_ok=True)

    columns = best["features"]
    model = clone(best["model"])
    model.fit(X_dev[columns], y_dev)

    print("\n[B] Cross-session test — per-class window predictions (batched)")
    print(f"  Model: {best['model_name']}  |  Features: {best['variant']}")
    print(
        f"  {'Class':<12} {'Windows':>7}  {'Correct':>7}  {'Acc':>7}  "
        f"{'Prediction breakdown'}"
    )

    all_true = []
    all_pred = []

    for label in sorted(cross_data):
        df = cross_data[label]
        X_cs, _, _ = build_feature_matrix(df)
        if X_cs.empty:
            print(f"  Class {label} ({CLASS_NAMES[label]}): "
                  f"no complete windows (need >= {WINDOW_SIZE} rows)")
            continue

        X_cs = X_cs.reindex(columns=columns, fill_value=0)
        preds = model.predict(X_cs)
        correct = int(np.sum(preds == label))
        acc_cls = correct / len(preds) * 100

        pred_counts = {
            CLASS_NAMES[l]: int(np.sum(preds == l))
            for l in sorted(CLASS_NAMES)
            if np.sum(preds == l) > 0
        }
        dist_str = "  ".join(f"{k}={v}" for k, v in pred_counts.items())
        print(
            f"  {CLASS_NAMES[label]:<12} {len(preds):>7}  {correct:>7}  "
            f"{acc_cls:>6.1f}%  {dist_str}"
        )
        all_true.extend([label] * len(preds))
        all_pred.extend(preds.tolist())

    if not all_true:
        print("  No windows available for cross-session evaluation.")
        return

    all_true = np.array(all_true)
    all_pred = np.array(all_pred)
    overall_acc = accuracy_score(all_true, all_pred) * 100
    overall_f1 = f1_score(all_true, all_pred, average="macro", zero_division=0) * 100
    print(
        f"\n  Overall cross-session -> "
        f"Acc={overall_acc:.1f}%   macro-F1={overall_f1:.1f}%"
    )

    # ── Detailed text report ──────────────────────────────────────────────────
    print("\n[C] Cross-session classification report (available classes only)...")
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

    print("  Text confusion matrix (cross-session):")
    print_confusion_matrix(all_true, all_pred, labels_present)

    # ── Matplotlib confusion matrix ───────────────────────────────────────────
    print("\n[E] Saving confusion matrix figures...")
    cm = confusion_matrix(all_true, all_pred, labels=labels_present)
    label_names = [CLASS_NAMES[l] for l in labels_present]

    # (i) Batched cross-session figure
    cs_fig_path = os.path.join(output_dir, "confusion_matrix_cross_session.png")
    _save_confusion_matrix_figure(
        cm=cm,
        labels=label_names,
        title="Cross-Session Confusion Matrix",
        filepath=cs_fig_path,
        acc=overall_acc,
        macro_f1=overall_f1,
        model_name=best["model_name"],
        feature_variant=best["variant"],
    )

    # ── Signal statistics comparison ──────────────────────────────────────────
    print("\n[D] Raw signal statistics — Session A vs Cross-session (per class)")
    header = f"  {'Class':<12}  {'Metric':<20}  {'Session A':>11}  {'Cross-session':>13}"
    print(header)
    print("  " + "-" * 62)

    for label in sorted(cross_data):
        cs_df = cross_data[label]
        full_path = os.path.join(DATA_DIR, FILE_MAP[label])
        sa_raw = pd.read_csv(
            full_path, header=None, names=COLS, on_bad_lines="skip", encoding="utf-8"
        )
        for c in COLS:
            sa_raw[c] = pd.to_numeric(sa_raw[c], errors="coerce")
        sa_df = sa_raw.dropna(subset=["timestamp_ms", "ina_a_ma"])
        sa_df = sa_df[sa_df["ina_a_ma"].abs() >= DEAD_CURRENT_THRESHOLD]

        rms_stds_sa = [
            np.std(sa_df["accel_rms"].values[s: s + WINDOW_SIZE])
            for s in range(0, len(sa_df) - WINDOW_SIZE + 1, STEP_SIZE)
        ]
        rms_stds_cs = [
            np.std(cs_df["accel_rms"].values[s: s + WINDOW_SIZE])
            for s in range(0, len(cs_df) - WINDOW_SIZE + 1, STEP_SIZE)
        ]

        stats = [
            ("cur mean (mA)",    sa_df["ina_a_ma"].mean(),      cs_df["ina_a_ma"].mean()),
            ("cur std (mA)",     sa_df["ina_a_ma"].std(),       cs_df["ina_a_ma"].std()),
            ("pwr mean (mW)",    sa_df["ina_a_mw"].mean(),      cs_df["ina_a_mw"].mean()),
            ("accel_rms mean",   sa_df["accel_rms"].mean(),     cs_df["accel_rms"].mean()),
            ("accel_rms_std/win",np.mean(rms_stds_sa) if rms_stds_sa else float("nan"),
                                 np.mean(rms_stds_cs) if rms_stds_cs else float("nan")),
        ]
        first = True
        for metric, sa_val, cs_val in stats:
            cls_label = CLASS_NAMES[label] if first else ""
            print(f"  {cls_label:<12}  {metric:<20}  {sa_val:>11.3f}  {cs_val:>13.3f}")
            first = False
        print("  " + "-" * 62)



# ─── LOSO ─────────────────────────────────────────────────────────────────────
def run_loso(output_dir):
    """
    2-fold Leave-One-Session-Out cross-validation.
      Fold 1: Train = Session A  ->  Test = Session B
      Fold 2: Train = Session B  ->  Test = Session A

    Within each fold, purged blocked CV is run on the training session to
    select the best model and feature variant — no information from the test
    session leaks into that selection.
    """
    os.makedirs(output_dir, exist_ok=True)

    print("\n" + "=" * 72)
    print("  Leave-One-Session-Out (LOSO) — 2-Fold Cross-Session Evaluation")
    print("=" * 72)

    fold_metrics = []

    for fold_num, (train_id, test_id) in enumerate([("A", "B"), ("B", "A")], start=1):
        print(f"\n{'─' * 72}")
        print(f"  Fold {fold_num}:  Train = Session {train_id}   |   Test = Session {test_id}")
        print(f"{'─' * 72}")

        # ── Load training session ─────────────────────────────────────────────
        print(f"\n  Loading Session {train_id} (training)...")
        train_raw = {}
        for label, (fp, enc, keep) in SESSION_MAP[train_id].items():
            train_raw[label] = load_session_file(fp, label, enc, keep)

        # ── Load test session ─────────────────────────────────────────────────
        print(f"\n  Loading Session {test_id} (test)...")
        test_raw = {}
        for label, (fp, enc, keep) in SESSION_MAP[test_id].items():
            test_raw[label] = load_session_file(fp, label, enc, keep)

        # ── Feature extraction on full sessions (no within-session split) ─────
        X_train, y_train, train_meta = build_dataset(train_raw)
        X_test,  y_test,  _          = build_dataset(test_raw)
        print(
            f"\n  Training matrix: {X_train.shape}  |  "
            f"Test matrix: {X_test.shape}"
        )
        for lbl in sorted(SESSION_MAP[train_id]):
            print(
                f"    Class {lbl}: "
                f"{np.sum(y_train == lbl)} train windows, "
                f"{np.sum(y_test  == lbl)} test windows"
            )

        # ── Feature / model selection via purged blocked CV on training session
        feature_variants = get_feature_variants(X_train.columns)
        splits = make_purged_blocked_splits(train_meta, train_raw)
        results = evaluate_development(
            X_train, y_train, splits, feature_variants,
            header=f"  Feature selection (purged CV on Session {train_id})..."
        )
        best = choose_best_result(results)
        print(
            f"\n  Best: {best['model_name']} / {best['variant']}  "
            f"(CV macro-F1={best['cv_macro_f1'] * 100:.1f}%)"
        )

        # ── Train on full training session, predict on test session ───────────
        model = clone(best["model"])
        cols  = best["features"]
        model.fit(X_train[cols], y_train)
        X_test_aligned = X_test.reindex(columns=cols, fill_value=0)
        y_pred = model.predict(X_test_aligned)

        # ── Metrics ───────────────────────────────────────────────────────────
        labels_present = sorted(np.unique(y_test))
        target_names   = [CLASS_NAMES[l] for l in labels_present]
        acc = accuracy_score(y_test, y_pred) * 100
        f1  = f1_score(y_test, y_pred, average="macro", zero_division=0) * 100

        print(f"\n  Fold {fold_num} result -> Acc={acc:.1f}%   Macro-F1={f1:.1f}%\n")
        print(
            classification_report(
                y_test, y_pred,
                labels=labels_present,
                target_names=target_names,
                zero_division=0,
            )
        )
        print("  Confusion Matrix:")
        print_confusion_matrix(y_test, y_pred, labels_present)

        # ── Save confusion matrix figure ──────────────────────────────────────
        cm       = confusion_matrix(y_test, y_pred, labels=labels_present)
        fig_path = os.path.join(
            output_dir, f"confusion_matrix_loso_fold{fold_num}.png"
        )
        _save_confusion_matrix_figure(
            cm=cm,
            labels=target_names,
            title=(
                f"LOSO Fold {fold_num}  —  "
                f"Train: Session {train_id}  |  Test: Session {test_id}"
            ),
            filepath=fig_path,
            acc=acc,
            macro_f1=f1,
            model_name=best["model_name"],
            feature_variant=best["variant"],
        )

        fold_metrics.append(
            {"fold": fold_num, "train": train_id, "test": test_id,
             "acc": acc, "f1": f1}
        )

    # ── LOSO summary ─────────────────────────────────────────────────────────
    print("\n" + "=" * 72)
    print("  LOSO Summary")
    print("=" * 72)
    print(
        f"\n  {'Fold':<8} {'Train':>9} {'Test':>8}  "
        f"{'Accuracy':>9}  {'Macro-F1':>9}"
    )
    print("  " + "-" * 48)
    for m in fold_metrics:
        print(
            f"  Fold {m['fold']:<3}   Session {m['train']:>1}   Session {m['test']:>1}   "
            f"{m['acc']:>8.1f}%  {m['f1']:>8.1f}%"
        )
    avg_acc = np.mean([m["acc"] for m in fold_metrics])
    avg_f1  = np.mean([m["f1"]  for m in fold_metrics])
    print("  " + "-" * 48)
    print(
        f"  {'LOSO avg':<8} {'':>9} {'':>8}   "
        f"{avg_acc:>8.1f}%  {avg_f1:>8.1f}%"
    )
    print(
        f"\n  Interpretation:\n"
        f"    LOSO average is the best available estimate of real-world\n"
        f"    cross-session performance with only 2 sessions.\n"
        f"    A final production model should train on ALL data (A + B combined)."
    )


# ─── MAIN ─────────────────────────────────────────────────────────────────────
def main(mode="cross_session"):
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

    # ── Export trained model for bridge/inference use ─────────────────────────
    import pickle
    model_path = os.path.join(BASE_DIR, "model.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(
            {
                "model": best_model,           # already fitted on full dev set
                "feature_cols": best["features"],
                "class_names": CLASS_NAMES,    # {1: "unloaded", 2: ..., 4: "stall"}
                "window_size": WINDOW_SIZE,
                "step_size": STEP_SIZE,
                "variant": best["variant"],
                "model_name": best["model_name"],
            },
            f,
        )
    print(f"\n  [✓] Model exported -> {model_path}")
    print(f"      Variant : {best['variant']}  ({len(best['features'])} features)")
    print(f"      Model   : {best['model_name']}")

    embedded_best = choose_best_random_forest_result(results)
    embedded_model = clone(embedded_best["model"])
    embedded_model.fit(X_dev[embedded_best["features"]], y_dev)
    firmware_dir = os.path.abspath(os.path.join(BASE_DIR, "..", "firmware"))
    header_path, source_path = export_random_forest_firmware(
        embedded_model,
        embedded_best["features"],
        CLASS_NAMES,
        firmware_dir,
    )
    print("\n  [✓] ESP32 Random Forest export")
    print(f"      Header  : {header_path}")
    print(f"      Source  : {source_path}")
    print(
        f"      Variant : {embedded_best['variant']}  "
        f"({len(embedded_best['features'])} features)"
    )

    if mode == "cross_session":
        # ── Cross-session evaluation (Original ~77% run) ───────────────────────
        cross_data = load_cross_session_data()
        evaluate_cross_session(best, X_dev, y_dev, cross_data, OUTPUT_DIR)

        print("\n" + "=" * 72)
        print("  Pipeline complete.")
        print(
            "  Note: within-session test = later data from same recordings.\n"
            f"        Cross-session test  = separate recording (different session).\n"
            f"        Figures saved in    -> {OUTPUT_DIR}/"
        )
        print("=" * 72)
    else:
        # ── LOSO cross-session evaluation (Rigorous ~65% run) ──────────────────
        run_loso(OUTPUT_DIR)

        print("\n" + "=" * 72)
        print("  Pipeline complete.")
        print(
            "  Steps [1]-[7]: within-session baseline (Session A only).\n"
            f"  LOSO section:  2-fold cross-session evaluation.\n"
            f"  Figures saved in -> {OUTPUT_DIR}/"
        )
        print("=" * 72)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the NILM pipeline")
    parser.add_argument(
        "--mode",
        choices=["cross_session", "loso"],
        default="cross_session",
        help="Evaluation mode: 'cross_session' (A->B only, ~77%) or 'loso' (2-fold LOSO, ~65%)"
    )
    args = parser.parse_args()
    main(mode=args.mode)
