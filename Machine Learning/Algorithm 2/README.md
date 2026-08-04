# Algorithm 2 — Classical ML with Spatial Coordinates

## Overview

Algorithm 2 extends Algorithm 1 by adding the spatial X/Y position of each scan
point as additional features. Rather than averaging a patient's measurements into
one number, it classifies **individual scan points** using three features:
Brillouin plateau shift, X position, and Y position on the cornea surface.

This allows the model to potentially learn that the spatial location of a measurement
matters for diagnosis — specifically that central points (near X=0, Y=0) may carry
more diagnostic information than peripheral ones, consistent with the focal nature
of keratoconus.

**Key result:** Best point-level Leave-One-Patient-Out accuracy is **84.3% (SVM)**,
notably lower than Algorithm 1's 90–97% patient-level accuracy. This gap is
expected and explained: individual scan points are significantly noisier than
patient-averaged values (per-point SNR ~1.48 vs ~8.91 after averaging). The result
is not directly comparable to Algorithm 1 without a majority-vote aggregation step
to convert point predictions back to a patient-level diagnosis.

---

## Files

```
plateauXY.py              Data pipeline — reads per-patient Excel files,
                          extracts Plateau + X/Y coordinates, outputs
                          combined_data.xlsx

algorithimTrainingXY.py   ML pipeline — reads combined_data.xlsx, trains
                          11 classifiers on 3 features, evaluates with
                          both single split and LOGO cross-validation
```

---

## Run Order

```bash
python plateauXY.py              # produces combined_data.xlsx
python algorithimTrainingXY.py   # reads combined_data.xlsx, prints results
```

---

## File Details

### plateauXY.py — Data Preparation with Coordinates

**What it does**

Same job as `dataAutomation.py` (Algorithm 1) but extracts three columns per scan
point instead of one: the Brillouin plateau shift, the X coordinate, and the Y
coordinate of that point on the cornea surface.

**Key difference from Algorithm 1's data script**

Most patient files store Plateau values in a `Brillouin data` sheet and X/Y
coordinates in a separate `XY positions` sheet. This script reads both sheets per
file and joins them by row position (row N of Plateau corresponds to row N of XY).
One patient (20230124) stores both in the same `Sheet1`.

**Input files required**

| File | Description |
|---|---|
| Per-patient `.xlsx` files | Must contain a `Brillouin data` sheet (Plateau) and an `XY positions` sheet (X mm, Y mm) |
| `SKC_names.xlsx` | Diagnosis key: Controls and SKC patient ID lists |

**Key configuration**

```python
FOLDER_PATH = '/Users/minhnguyen/try-scipy/Minh Data/Patient Data/*.xlsx'
SKC_FILE    = '/Users/minhnguyen/try-scipy/Minh Data/SKC_names.xlsx'

# Files with non-standard formatting
OUTLIERS = {
    'Metrics at selected points 20230124.xlsx': ('Sheet1', 'Brillouin shifts of plateau before'),
    'Metrics at selected points 20230420.xlsx': ('Brillouin data', 'Plateau before'),
}
```

**Output**

`combined_data.xlsx` with five columns:

| Column | Description |
|---|---|
| Patient | Patient identifier string |
| Plateau | Brillouin plateau shift (GHz) at this scan point |
| X (mm) | Horizontal position on cornea surface in mm |
| Y (mm) | Vertical position on cornea surface in mm |
| Diagnosis | `Controls`, `SKC`, or `Unknown` |

Approximately 1,081 rows across 30 patients. This file is also used as the
coordinate mapping reference by the deep learning pipeline (`2mm.py`).

---

### algorithimTrainingXY.py — ML Classification with Spatial Features

**What it does**

Reads `combined_data.xlsx` and trains eleven classifiers on the three-feature
representation (Plateau, X, Y) at the individual scan point level. Evaluates with
both a single 80/20 patient split (diagnostic) and Leave-One-Group-Out
cross-validation (primary result).

**Three features**

```
Plateau  — Brillouin frequency shift (stiffness) at this spatial location (GHz)
X (mm)   — horizontal position of this scan point on the cornea surface
Y (mm)   — vertical position of this scan point on the cornea surface
```

Radial distance from center = √(X² + Y²). Points within 2mm of center have the
highest diagnostic relevance per the published literature and the spatial analysis
performed during this project (Cohen's d = 2.31 at center vs 1.78 at periphery).

**Patient-level splitting is critical**

Each patient has ~30–44 scan points. A standard random split on rows would put the
same patient's points on both sides of the train/test boundary, allowing the model
to recognize patient-specific signal levels instead of learning a generalizable
rule. `GroupShuffleSplit` with `groups=Patient` ensures every patient is wholly in
training or wholly in test.

**The eleven classifiers**

| Name | Algorithm | Scaling | Notes |
|---|---|---|---|
| LR | Logistic Regression | Yes | C=1.0, balanced |
| LDA | Linear Discriminant Analysis | No | No hyperparameters |
| KNN | K-Nearest Neighbors | No | n_neighbors=3 — unscaled, distance biased |
| CART | Decision Tree | No | max_depth=3, min_samples_leaf=2 |
| NB | Naive Bayes | No | Gaussian |
| SVM | Support Vector Machine | Yes | RBF kernel, gamma=scale, C=1.0 |
| RF | Random Forest | No | 100 trees, max_depth=4 |
| GB | Gradient Boosting | No | max_depth=3 |
| ET | Extra Trees | No | 100 trees, max_depth=4 |
| Ridge | Ridge Classifier | No | alpha=1.0 — unscaled |
| Bag | Bagging | No | 25 estimators, max_samples=0.8 |

> **Note:** KNN and Ridge are not wrapped in a StandardScaler pipeline in this
> script. Plateau values (~3.7–4.7 GHz) and X/Y coordinates (~-4 to +4 mm) are
> on different numeric scales. This will bias KNN toward whichever feature has
> larger raw values, and may reduce Ridge's ability to learn optimal coefficients.
> This is preserved as-is from the original code.

**Two evaluation methods**

*Single Split (80/20):* approximately 24 patients train, 6 patients test. A fast
diagnostic for overfitting. The train-vs-validation gap interpretation:

| Gap | Interpretation |
|---|---|
| < 10% | Reasonable generalization on this split |
| 10–20% | Caution, some overfitting |
| > 20% | Significant overfitting |
| Negative | Validation set was accidentally easier than training |

This is not the primary result. With only ~6 test patients, one wrong prediction
is worth 16.7% accuracy and results vary heavily with which patients land in the
test set.

*Leave-One-Group-Out CV (primary):* holds out all of one patient's scan points,
trains on everyone else's, repeats for all 30 patients. The accuracy reported is
the fraction of **individual scan points** correctly classified across all folds,
not the fraction of patients correctly diagnosed.

> **Important:** LOGO accuracy here is a point-level metric. A patient with 36
> scan points contributes 36 individual predictions to the overall accuracy count.
> This is a harder task than Algorithm 1's patient-level classification and the two
> numbers are not directly comparable without aggregating point predictions back
> to the patient level via majority vote.

**Key configuration**

```python
COMBINED_FILE = '/Users/minhnguyen/try-scipy/combined_data.xlsx'
# GroupShuffleSplit parameters
train_size  = 0.8
random_state = 3   # different from Algorithm 1 (seed=3 vs seed=1)
```

**Results**

Best Leave-One-Group-Out accuracy: **84.3% (SVM)**. Most linear models (LR, LDA,
Ridge) cluster around 75%. KNN drops to 56.3% — consistent with the unscaled
feature problem noted above.

The gap between SVM's 84.3% and the linear models' ~75% suggests a nonlinear
relationship between spatial position and diagnosis that the RBF kernel can capture
but linear boundaries cannot.

---

## Data Files Required

| File | Description |
|---|---|
| Per-patient `.xlsx` files | Must have both `Brillouin data` and `XY positions` sheets |
| `SKC_names.xlsx` | Diagnosis key |
| `combined_data.xlsx` | Output of `plateauXY.py`, input to `algorithimTrainingXY.py` |

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

Algorithm 2 sits between Algorithm 1 (patient-averaged, no spatial info) and the
deep learning pipelines (raw depth profiles, full temporal structure):

- **Algorithm 1** averages all points per patient → one number per patient → 90–97%
  patient-level accuracy. Simpler, more interpretable, stronger result.
- **Algorithm 2** keeps all points with X/Y coordinates → three features per point
  → 84.3% point-level accuracy. Harder task due to lower per-point SNR.
- **Deep learning** (`buildDataSet.py`, `segmentation.py`): uses the full 100-frame
  depth profile per point. `combined_data.xlsx` produced by `plateauXY.py` is also
  used by `2mm.py` as the coordinate mapping for the 2mm spatial filter.

---

## Radial Separability Analysis (Key Finding)

A radial separability analysis was performed to test whether peripheral scan points
carry meaningful diagnostic signal or are essentially noise. Results:

| Radius bin | Cohen's d | p-value | Point accuracy |
|---|---|---|---|
| 0–1mm (center) | 2.31 | 2.35×10⁻²¹ | 90.5% |
| 1–2mm | 1.55 | 6.53×10⁻²⁷ | 79.7% |
| 2–3mm | 1.24 | 3.34×10⁻¹⁸ | 74.5% |
| 3mm+ (periphery) | 1.78 | 3.65×10⁻³⁶ | 81.6% |

Every radius bin shows large, statistically significant separation between Controls
and SKC. The periphery is not noise. This finding was used to inform the spatial
filtering strategy in the deep learning pipeline.

---

*Developed as part of a Brillouin microscopy internship at the University of
Maryland Fischell Department of Bioengineering under the supervision of Ron and
Giuliano Scarcelli. Dataset: 30 patients (15 Controls, 15 SKC). Reference:
Zhang et al., American Journal of Ophthalmology, 2023.*
