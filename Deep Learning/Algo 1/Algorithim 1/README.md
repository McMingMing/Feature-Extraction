# Brillouin Cornea Classification — .mat File Pipeline

## Overview

This pipeline classifies corneal tissue as **Controls** (healthy) or **Subclinical
Keratoconus (SKC)** using Lorentzian-fitted Brillouin frequency shift data stored in
`.mat` files. It contains two parallel approaches that run on the same underlying
data: a classical **Machine Learning** pipeline and an experimental **Deep Learning
(LSTM)** pipeline.

Both pipelines use the same 30 patients (15 Controls, 15 SKC). Each patient has one
`shifts_before.mat` file containing a 100 × N array, where each of the N columns is
one spatial scan point on the cornea and each of the 100 rows is one depth frame.
The coordinate mapping (which column corresponds to which X/Y position on the cornea)
comes from `combined_data.xlsx`.

---

## File Structure

```
masterdataSet.py      ML Step 1 — reads combined_data.xlsx, engineers spatial
                      features per patient, writes ML_Master_Dataset.xlsx

classifiers.py        ML Step 2 — reads ML_Master_Dataset.xlsx, trains 8
                      classifiers, reports accuracy with confidence intervals

buildDataSet.py       DL Step 1 — reads shifts_before.mat files for all 30
                      patients, extracts per-point depth profiles, saves
                      dl_dataset.npz

runLTSM.py            DL Step 2 — reads dl_dataset.npz, trains a stacked
                      LSTM classifier, evaluates at both sequence and
                      patient level
```

---

## Run Order

**Machine Learning:**
```bash
python masterdataSet.py    # produces ML_Master_Dataset.xlsx
python classifiers.py      # reads ML_Master_Dataset.xlsx, prints results
```

**Deep Learning:**
```bash
python buildDataSet.py     # produces dl_dataset.npz
python runLTSM.py          # reads dl_dataset.npz, trains and evaluates LSTM
```

---

## Machine Learning Pipeline

### masterdataSet.py — Feature Engineering

**What it does**

Reads `combined_data.xlsx` (one row per spatial scan point per patient, with Plateau
value and X/Y coordinates) and collapses each patient's many scan points into a
single row of engineered features. The output is `ML_Master_Dataset.xlsx` with one
row per patient.

**Why this design**

The raw data has ~30 to 44 scan points per patient. Feeding each point as a separate
training sample would create data leakage (the same patient's points appearing on
both sides of a train/test split). Instead, each patient is summarized into five
spatial features that capture both the overall stiffness level and its spatial
distribution across the cornea.

**The five engineered features**

| Feature | Description | Biological meaning |
|---|---|---|
| `Mean_Plateau_2mm` | Mean plateau of all points within 2mm of cornea center | Overall central stiffness — the strongest single discriminator |
| `Mean_Plateau_All` | Mean plateau of all points regardless of location | Whole-cornea average stiffness |
| `Std_Plateau_2mm` | Standard deviation of plateau within 2mm | Uniformity — healthy corneas are spatially uniform; KC corneas are not |
| `Center_Periphery_Gradient` | Mean(inner ring < 1mm) − Mean(middle ring 1–2mm) | Spatial stiffness gradient — negative in KC where the center is softer |
| `Min_Plateau_2mm` | Minimum plateau value within 2mm | Localized soft spot — captures the KC cone |

**Key configuration options**

```python
INPUT_FILE  = '/Users/minhnguyen/try-scipy/combined_data.xlsx'
OUTPUT_FILE = '/Users/minhnguyen/try-scipy/ML_Master_Dataset.xlsx'
INNER_RADIUS  = 1.0   # mm — inner ring boundary
REGION_RADIUS = 2.0   # mm — outer boundary of the region of interest
```

---

### classifiers.py — Classifier Comparison

**What it does**

Reads `ML_Master_Dataset.xlsx` and trains eight classifiers on the engineered
features. Reports accuracy under two evaluation methods: 5-fold stratified
cross-validation and Leave-One-Patient-Out cross-validation. The Leave-One-Out
result is the primary number to report.

**Feature sets**

The script supports four interchangeable feature sets controlled by `ACTIVE_SET`:

```python
FEATURE_SETS = {
    'mentor_2mm_only': ['Mean_Plateau_2mm'],                          # 1 feature
    'mentor_with_avg': ['Mean_Plateau_2mm', 'Mean_Plateau_All'],      # 2 features
    'spatial':         ['Mean_Plateau_2mm', 'Std_Plateau_2mm',
                        'Center_Periphery_Gradient'],                  # 3 features
    'all_features':    ['Mean_Plateau_2mm', 'Mean_Plateau_All',
                        'Std_Plateau_2mm', 'Center_Periphery_Gradient',
                        'Min_Plateau_2mm'],                            # 5 features
}
ACTIVE_SET = 'all_features'   # change this to test different combinations
```

**The eight classifiers**

| Name | Algorithm | Notes |
|---|---|---|
| LR | Logistic Regression | Linear boundary, C=0.5 (high regularization) |
| Ridge | Ridge Classifier | Linear with L2 penalty, alpha=2.0 |
| SVM | Support Vector Machine | Linear kernel, C=0.5 |
| KNN | K-Nearest Neighbors | 5 neighbors |
| NB | Naive Bayes | Gaussian, no hyperparameters |
| CART | Decision Tree | max_depth=2 to prevent memorization |
| RF | Random Forest | 50 trees, max_depth=2 |
| ET | Extra Trees | 50 trees, max_depth=2 |

All distance-based and regularization-based models (LR, Ridge, SVM, KNN) are
wrapped in a StandardScaler pipeline. Plateau values are in GHz (~3.7–4.7 plotted
scale) and coordinates are in mm — scaling prevents the model from over-weighting
any one feature due to its raw numeric magnitude.

**Evaluation methods**

*5-Fold Stratified CV* splits the 30 patients into 5 groups of 6. The model trains
on 24 patients and tests on 6, repeated 5 times. The average accuracy and its
standard deviation are reported. With 6 patients per test fold, one wrong prediction
is a 16.7% swing, so high standard deviations are expected and normal.

*Leave-One-Patient-Out CV* trains on 29 patients and tests on 1, repeated 30 times
so every patient is tested exactly once. This is the most honest evaluation method
at this sample size and is the number to report. A 95% confidence interval is also
computed using the normal approximation: because N=30, the CI is wide (~±7 to 10
percentage points) and should always be reported alongside the headline accuracy.

**Results summary**

Using `all_features`, most classifiers achieve 90–97% Leave-One-Patient-Out
accuracy. Notably, `mentor_2mm_only` (a single feature: central mean stiffness)
matches the full five-feature model, confirming that the average Brillouin plateau
within the central 2mm radius is the dominant discriminating signal.

---

## Deep Learning Pipeline (LSTM)

### buildDataSet.py — Sequence Dataset Builder

**What it does**

Walks all 30 patient folders, reads each `shifts_before.mat` file, and extracts
every spatial point's depth profile as a separate training sequence. Saves the
result to `dl_dataset.npz`.

**Data structure**

Each `shifts_before.mat` is a 100 × N array. Each column is one spatial scan point.
Each row is one depth frame. So column 5 of patient 20211014's mat file is a
100-frame sequence representing how the Brillouin shift varies with depth at that
specific spatial location on the cornea. This is the input to the LSTM.

**Output arrays saved in dl_dataset.npz**

| Array | Shape | Contents |
|---|---|---|
| `X` | (n_sequences, 100, 1) | Normalized depth-profile sequences |
| `y` | (n_sequences,) | Binary label: 0=Controls, 1=SKC |
| `groups` | (n_sequences,) | Patient ID string for each sequence |

The `groups` array is critical. It tracks which patient every sequence came from so
the training script can split by patient rather than by sequence, preventing identity
leakage.

**Key configuration options**

```python
DATA_DIR       = '/Users/minhnguyen/deep-learning/All Brillouin Point Data'
DIAGNOSIS_FILE = '/Users/minhnguyen/deep-learning/combined_data.xlsx'
MAT_FILENAME   = 'shifts_before.mat'
OUTPUT_NPZ     = '/Users/minhnguyen/deep-learning/dl_dataset.npz'
```

**Diagnosis lookup**

The diagnosis for each patient is resolved by matching the patient folder name
against `combined_data.xlsx`. If the full folder name (e.g. `20220628 Left`)
does not match, the script tries the date prefix alone (`20220628`) to handle
naming edge cases. Any folder with no usable diagnosis is skipped and listed
in the summary.

---

### runLTSM.py — LSTM Classifier

**What it does**

Loads `dl_dataset.npz`, trains a stacked LSTM on the 100-frame depth profiles,
and evaluates performance at both the individual sequence level and the patient level.

**Architecture**

The model follows the structure from Ron's template (stacked LSTM layers → dense
output), adapted for this binary classification task:

```
Input: (100, 1)  — 100-frame normalized depth profile, 1 value per frame
  LSTM(64 units, return_sequences=True, dropout=0.2, L2 recurrent regularization)
  LSTM(64 units, return_sequences=True, dropout=0.2, L2 recurrent regularization)
  LSTM(64 units, dropout=0.2, L2 recurrent regularization)
  Dense(1, sigmoid)  — binary output: 0=Controls, 1=SKC
```

Ron's original template used 192 units per layer (12 × batch_size). This is reduced
to 64 because three 192-unit LSTM layers on ~1,000 short sequences will memorize
the training set. Increase `UNITS` to reproduce his exact width, but monitor the
train-vs-validation accuracy gap — a gap above 20% indicates overfitting.

**Key configuration options**

```python
DATASET_NPZ = 'dl_dataset.npz'
UNITS       = 64     # LSTM units per layer
EPOCHS      = 50     # max training epochs
BATCH_SIZE  = 16
TEST_FRAC   = 0.2    # fraction of PATIENTS held out
```

**Patient-level split**

The train/test split uses `GroupShuffleSplit` on the `groups` array (patient IDs).
This guarantees every patient is wholly in training or wholly in testing — no
patient's sequences appear on both sides. The script asserts this with a hard check
that will crash if any leakage is detected.

Approximately 24 patients (869 sequences) train and 6 patients (212 sequences) test.

**Normalization**

Raw Brillouin shifts are in Hz (~3.7 × 10⁹ to 4.7 × 10⁹). LSTMs train poorly on
values of this magnitude. The script standardizes all sequences to zero mean and
unit variance using `StandardScaler`. The scaler is fit on training data only and
then applied to both training and test sets, preventing information from the test
set from influencing the normalization.

**Class weighting**

The dataset has 594 Controls sequences and 487 SKC sequences (after filtering).
Without correction, a model can minimize the loss by predicting Controls for every
input and still achieve ~55% accuracy without learning anything meaningful. Class
weights are set inversely proportional to class frequency so each SKC mistake costs
proportionally more than a Controls mistake.

**Two-level evaluation**

*Sequence level:* what fraction of the 212 individual test sequences were correctly
classified. This is a harder task because each sequence is one noisy 100-frame
depth profile from one spatial point.

*Patient level:* for each test patient, all their sequences are voted on (majority
vote) and the patient is diagnosed as whichever class the majority of their points
were predicted to be. This is the apples-to-apples comparison against the ML
pipeline's patient-level accuracy.

**Results and known limitations**

The LSTM validation accuracy stalled at 44.7% across all 50 epochs, with
patient-level accuracy of 33.3% — equivalent to a coin flip. This was diagnosed as
a signal-to-noise problem: the between-class signal at the individual sequence level
(0.036 GHz) is smaller than the within-class noise (0.024 GHz), giving a
signal-to-noise ratio of only 1.48. The ML pipeline achieves 90–97% by averaging
~36 sequences per patient first, which cuts noise by √36 ≈ 6×, raising the effective
SNR to 8.91. The LSTM, working on individual unaveraged sequences, cannot overcome
this noise without substantially more data. See `runCNN.py` and `segmentation.py`
for follow-on approaches that address this limitation.

---

## Dependencies

```
numpy
pandas
scipy
scikit-learn
tensorflow
openpyxl
```

Install with:
```bash
/Users/minhnguyen/dl-env/bin/pip install numpy pandas scipy scikit-learn tensorflow openpyxl
```

---

## Data Files Required

| File | Description |
|---|---|
| `combined_data.xlsx` | One row per scan point: Patient, Plateau, X (mm), Y (mm), Diagnosis |
| `shifts_before.mat` | Per-patient Lorentzian-fitted Brillouin shifts, 100 × N array |
| `ML_Master_Dataset.xlsx` | Output of masterdataSet.py, input to classifiers.py |
| `dl_dataset.npz` | Output of buildDataSet.py, input to runLTSM.py |

---

*Developed as part of a Brillouin microscopy internship at the University of
Maryland Fischell Department of Bioengineering under the supervision of Ron and
Giuliano Scarcelli. Dataset: 30 patients (15 Controls, 15 SKC). Reference:
Zhang et al., American Journal of Ophthalmology, 2023.*
