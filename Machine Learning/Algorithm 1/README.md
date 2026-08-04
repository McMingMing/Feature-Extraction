# Algorithm 1 — Classical ML, Patient-Averaged Features

## Overview

Algorithm 1 classifies corneal tissue as **Controls** (healthy) or **Subclinical
Keratoconus (SKC)** using the simplest possible feature: the average Brillouin
plateau shift across all of a patient's spatial scan points. It establishes the
performance baseline for the broader project before any spatial information or
deep learning is introduced.

**Key result:** A single feature (mean Brillouin plateau within 2mm of the cornea
center) achieves **93.3% Leave-One-Patient-Out accuracy** across most classifiers,
with the best models reaching 96.7%. This demonstrates that overall corneal
stiffness, measured by Brillouin microscopy and averaged across the central scan
region, is strongly predictive of SKC status.

---

## Files

```
dataAutomation.py       Data pipeline — reads per-patient Excel files,
                        assigns diagnoses, outputs combined_plateau.xlsx

algorithimTraining.py   ML pipeline — reads combined_plateau.xlsx, averages
                        per patient, runs 8 classifiers with LOOCV
```

---

## Run Order

```bash
python dataAutomation.py       # produces combined_plateau.xlsx
python algorithimTraining.py   # reads combined_plateau.xlsx, prints results
```

---

## File Details

### dataAutomation.py — Data Preparation

**What it does**

Walks a folder of per-patient Excel files, extracts the Brillouin plateau column
from each one, matches each file to its diagnosis (Controls or SKC) using a lookup
spreadsheet, and combines everything into one master file (`combined_plateau.xlsx`)
with one row per spatial scan point.

**Input files required**

| File | Description |
|---|---|
| Per-patient `.xlsx` files | Named `Metrics at selected points YYYYMMDD.xlsx`. Each contains a `Plateau` column in a `Brillouin data` sheet |
| `SKC_names.xlsx` | Two columns: Controls and SKC, listing which patient IDs belong to each group |

**Diagnosis matching**

Patient filenames are matched to the diagnosis key by a word-level algorithm that
handles edge cases like `20220628 Left` and `20220628 Right` being different
patients. If no match is found, the patient is labeled `Unknown` and excluded from
training. Two files have non-standard Excel formatting and are handled as special
cases via the `OUTLIERS` dictionary in the configuration.

**Key configuration**

```python
FOLDER_PATH = '/Users/minhnguyen/try-scipy/Minh Data/Patient Data/*.xlsx'
SKC_FILE    = '/Users/minhnguyen/try-scipy/Minh Data/SKC_names.xlsx'

# Files with non-standard sheet/column names
OUTLIERS = {
    'Metrics at selected points 20230124.xlsx': ('Sheet1', 'Brillouin shifts of plateau before'),
    'Metrics at selected points 20230420.xlsx': ('Brillouin data', 'Plateau before'),
}
```

**Output**

`combined_plateau.xlsx` with three columns:

| Column | Description |
|---|---|
| Patient | Date-based patient identifier string (e.g. `20211014`) |
| Plateau | Lorentzian-fitted Brillouin plateau shift (GHz) at this scan point |
| Diagnosis | `Controls`, `SKC`, or `Unknown` |

Approximately 1,081 rows total across 30 patients (~30–44 rows per patient).

---

### algorithimTraining.py — ML Classification

**What it does**

Reads `combined_plateau.xlsx`, averages all of each patient's plateau values into
one number, trains eight classifiers on that single feature, and evaluates each
with Leave-One-Patient-Out cross-validation.

**Why average first**

Each patient has ~30–44 individual scan points. Treating each point as a separate
training sample would cause data leakage: the same patient's points could appear
on both the training and test sides of a split, letting the model recognize a
patient's overall signal level rather than learning a generalizable rule. Averaging
to one number per patient eliminates this problem and raises the effective
signal-to-noise ratio by approximately 6× (√36 ≈ 6).

**The single feature**

```
Mean_Plateau — per-patient mean of all spatial Brillouin plateau readings
```

This single feature is sufficient to achieve 90–97% classification accuracy.
Adding spatial features (as Algorithm 2 does) does not meaningfully improve on
this baseline at N=30.

**Key configuration**

```python
INPUT_FILE  = 'combined_plateau.xlsx'
OUTPUT_FILE = 'Algorithm1_Master_Dataset.xlsx'
FEATURES    = ['Mean_Plateau']
```

**The eight classifiers**

| Name | Algorithm | Settings |
|---|---|---|
| LR | Logistic Regression | C=0.5, liblinear solver |
| Ridge | Ridge Classifier | alpha=2.0 |
| SVM | Support Vector Machine | linear kernel, C=0.5 |
| KNN | K-Nearest Neighbors | n_neighbors=5 |
| NB | Naive Bayes | Gaussian |
| CART | Decision Tree | max_depth=2, min_samples_leaf=3 |
| RF | Random Forest | 50 trees, max_depth=2 |
| ET | Extra Trees | 50 trees, max_depth=2 |

All distance-based and regularization-based models (LR, Ridge, SVM, KNN) are
wrapped in a `StandardScaler` pipeline. Regularization parameters are set
conservatively (C=0.5, alpha=2.0, max_depth=2) because with only 29 training
patients per fold, overfitting is a real risk.

**Validation: Leave-One-Patient-Out CV**

The script uses Leave-One-Out cross-validation where each fold holds out exactly
one patient, trains on the remaining 29, and tests on the held-out patient.
Repeating this for all 30 patients gives one accuracy estimate per patient. The
final reported accuracy is the fraction of those 30 predictions that were correct.

This is the most appropriate validation method at N=30 because:
- Standard random splits test on only ~6 patients (20%), making results highly
  dependent on which 6 happen to be chosen
- Every patient gets to be the test case exactly once
- The maximum amount of training data is used at each fold

**Reading the output**

```
LR   :  90.0%  (27/30 correct)  approx 95% CI 79.3%-100.0%
```

- `90.0%` — 27 out of 30 patients correctly classified
- `27/30 correct` — explicit count for clarity
- `approx 95% CI 79.3%-100.0%` — the plausible range of true accuracy on new
  unseen patients. With N=30, one wrong prediction swings accuracy by 3.3
  percentage points, so CIs are wide (~±10%). Always report the CI alongside
  the headline number rather than treating 90.0% as a precise value.

**Results**

Most classifiers converge near **90.0%** (27/30 correct). CART occasionally
reaches 96.7% (29/30) but this reflects one extra correct prediction at this
sample size, not a meaningful advantage over the other models. Treat
model-to-model differences of a few percent as noise at N=30.

The consistency across eight very different model types (linear, distance-based,
probabilistic, tree-based) confirms the 90% result is a property of the data's
separability, not an artifact of any particular algorithm.

**Output**

`Algorithm1_Master_Dataset.xlsx` — one row per patient with columns:
Patient, Mean_Plateau, Diagnosis. Used for manual inspection and as a reference
for comparing against Algorithm 2 and the deep learning results.

---

## Data Files Required

| File | Description |
|---|---|
| Per-patient `.xlsx` files | One per patient, in `FOLDER_PATH` |
| `SKC_names.xlsx` | Diagnosis key |
| `combined_plateau.xlsx` | Output of `dataAutomation.py`, input to `algorithimTraining.py` |

---

## Dependencies

```
pandas
numpy
scikit-learn
openpyxl
```

Install with:
```bash
pip install pandas numpy scikit-learn openpyxl
```

---

## Relationship to Other Pipelines

Algorithm 1 is the simplest and most interpretable approach in the project.
Its 90–97% accuracy using a single averaged feature sets the performance ceiling
that every subsequent method is compared against:

- **Algorithm 2** (`plateauXY.py` + `algorithimTrainingXY.py`): adds spatial X/Y
  coordinates as additional features, classifying individual scan points rather
  than patient averages. Achieves ~84% point-level accuracy but is not directly
  comparable to Algorithm 1's patient-level result without majority voting.
- **Deep learning pipelines** (`buildDataSet.py`, `runLTSM.py`, `segmentation.py`):
  operate on raw depth profiles rather than aggregated features. The segmentation
  pipeline recovers a statistically significant Controls vs SKC difference
  (p < 0.0001) but with wider box plot distributions than Algorithm 1.

---

*Developed as part of a Brillouin microscopy internship at the University of
Maryland Fischell Department of Bioengineering under the supervision of Ron and
Giuliano Scarcelli. Dataset: 30 patients (15 Controls, 15 SKC). Reference:
Zhang et al., American Journal of Ophthalmology, 2023.*
