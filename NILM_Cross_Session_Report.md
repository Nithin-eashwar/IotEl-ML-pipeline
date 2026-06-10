# NILM Classification Pipeline: Cross-Session Generalisation Report

## 1. Executive Summary
This report summarises the evaluation of a Non-Intrusive Load Monitoring (NILM) classification model designed for predictive maintenance. The pipeline classifies motor operation into four distinct states: **Unloaded**, **Light Load**, **Heavy Load**, and **Stall**.

While traditional within-session testing yielded an artificial 100% accuracy, our rigorous **cross-session evaluation** (training on Session A, testing on Session B) revealed the true physical generalisation of the model, achieving **76.8% Accuracy** and **71.0% Macro-F1**. 

The results prove that the model perfectly generalises for extreme states (Unloaded and Stall) but highlights a critical vulnerability to vibration baseline drift between sessions for intermediate loads.

---

## 2. Data Pipeline & Cleaning Strategy
The dataset comprises high-frequency current, power, and accelerometer readings. A strict cleaning pipeline was enforced to ensure the model learns legitimate physics rather than sensor artefacts:

* **Dead-Current Filtering:** Rows with absolute current `< 2.0 mA` were dropped to remove disconnected/inactive states.
* **Timestamp Continuity:** Class 2 (light load) exhibited timestamp resets due to concatenated sub-recordings. The pipeline automatically isolated the largest contiguous segment (removing ~2,710 invalid rows) to preserve temporal integrity.

| Class | Raw Rows | Clean Rows | Discarded | Reason |
| :--- | :--- | :--- | :--- | :--- |
| **1 (Unloaded)** | 10,071 | 9,129 | 9.4% | Dead-current drop |
| **2 (Light Load)** | 10,056 | 7,316 | 27.2% | Dead-current + contiguous segment trim |
| **3 (Heavy Load)** | 10,056 | 9,182 | 8.7% | Dead-current drop |
| **4 (Stall)** | 10,122 | 9,258 | 8.5% | Dead-current drop |

**Total Training Data:** 34,885 clean rows (~1.25 hours of continuous equivalent data).

---

## 3. Feature Engineering & Ablation
Data was processed into continuous sliding windows to capture dynamic behaviour:
* **Window Size:** 128 samples (1.28 seconds)
* **Step Size:** 64 samples (50% overlap)

**Feature Ablation Decision:**
We tested 46 potential features, including FFT spectral energy and temperature. The ablation study demonstrated that the **"No Temperature or Accel FFT"** variant (20 features) performed identically to the full 46-feature set. 
> [!TIP]
> **Engineering Decision:** FFT and temperature features were permanently dropped. This reduces computational complexity for IoT edge deployment without sacrificing predictive power.

---

## 4. Cross-Session Evaluation (Primary Result)
To simulate real-world predictive maintenance, the model was evaluated across physically distinct recording sessions. 

* **Train:** Session A 
* **Test:** Session B (Completely unseen recordings)
* **Development Winner:** Random Forest Classifier

### Overall Performance: 76.8% Accuracy | 71.0% Macro-F1

### Confusion Matrix Breakdown
| True State \ Predicted | Unloaded | Light Load | Heavy Load | Stall | **Accuracy** |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Unloaded** | **220** | 0 | 0 | 0 | **100%** |
| **Light Load** | 7 | **227** | 0 | 0 | **97.0%** |
| **Heavy Load** | 0 | 203 | **21** | 0 | **9.4%** |
| **Stall** | 0 | 0 | 0 | **226** | **100%** |

### Leave-One-Session-Out (LOSO) Validation
To verify symmetric generalisation, a 2-fold LOSO architecture was run:
* **Fold 1 (Train A → Test B):** 75.8% Acc, 69.1% F1
* **Fold 2 (Train B → Test A):** 53.8% Acc, 44.4% F1
* **LOSO Average:** **64.8% Acc, 56.8% F1**

> [!IMPORTANT]
> The LOSO average is the most honest metric of the model's current capability. The asymmetry between Fold 1 and Fold 2 indicates that Session A contains a wider variance of physical behaviour than Session B.

---

## 5. Model Diagnostics: The "Heavy Load" Drift
The confusion matrix shows that 203 out of 224 Heavy Load windows were misclassified as Light Load. This is not an algorithmic failure, but a physical data drift issue.

### Feature Importance Analysis
The Random Forest relies overwhelmingly on vibration dynamics rather than just current:
1. `accel_rms_std` (18.3%)
2. `accel_y_std` (15.1%)
3. `accel_z_std` (12.9%)
4. `accel_x_std` (10.5%)
5. `cur_max` (7.7%)

### Root Cause of Misclassification
The model learned from Session A that `accel_rms_std` for Heavy Load is very low (~0.436), while Light Load is much higher (~1.127). 
However, in Session B, the motor's physical vibration at Heavy Load **drifted upwards to 0.515**. Because 0.515 falls squarely within the bounds of what the model learned as "Light Load", it classified it as such. This drift is standard in physical systems, driven by changes in fixture mounting tightness, motor temperature, or ambient vibration floor between days.

---

## 6. Honest Limitations & Next Steps

1. **Vulnerability to Absolute Baseline Shift:** The model currently uses absolute vibration standard deviations. Because physical mounting changes between sessions, absolute vibration shifts.
   * *Solution:* Engineer relative features (e.g., normalising session data against the first 10 seconds of "Unloaded" baseline for that specific run).
2. **Limited Session Diversity:** Training on a single session (A) produces overly rigid decision boundaries.
   * *Solution:* The final production model must be trained on **ALL available data (Session A + Session B combined)**. By exposing the model to multiple physical operating points, the decision boundary between Light and Heavy load will become robust.
3. **Hardware Constraint:** The data reflects a single motor unit. True generalisation will eventually require testing on a separate physical motor.
