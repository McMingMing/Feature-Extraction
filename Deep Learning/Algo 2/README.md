# Brillouin Cornea Segmentation — Deep Learning Pipeline

## Overview

This pipeline uses deep learning to automatically identify the corneal plateau
region in Brillouin depth profiles and measure the average stiffness of that
region. The goal is to replicate the manual plateau-selection process described
in Zhang et al. (AJO 2023) without human intervention, then use the automated
stiffness measurement to distinguish **Controls** (healthy) from **Subclinical
Keratoconus (SKC)** patients.

This is the deep learning component of the broader Brillouin cornea classification
project. It builds on the `.mat` file pipeline (see `README_mat_pipeline.md`) and
uses the same 30 patients and `shifts_before.mat` files.

---

## Files in This Pipeline

```
2mm.py               Dataset builder — reads all shifts_before.mat files,
                     optionally filters to points within 2mm of cornea
                     center, saves dl_dataset_2mm.npz

generateLabels.py    Label generator — produces per-frame binary cornea
                     labels (1=cornea, 0=other) for every depth profile,
                     saves labels_dataset.npz and verification plots

segmentation.py      1D U-Net — trains on the labeled dataset, predicts
                     cornea regions on new patients, computes stiffness
                     metrics, produces box plots comparing Controls vs SKC

cnn3.py              1D CNN classifier — alternative approach that classifies
                     depth profiles directly (without segmentation labels),
                     included for comparison against the segmentation pipeline
```

---

## Run Order

```bash
# Step 1: build the spatial dataset (all points or 2mm restricted)
python 2mm.py

# Step 2: generate binary cornea labels for segmentation training
python generateLabels.py

# Step 3: train the segmentation model and produce results
python segmentation.py

# Optional: run the CNN classifier for comparison
python cnn3.py
```

> **Note:** Run `2mm.py` before `generateLabels.py`. The label generator reads
> `shifts_before.mat` files directly, but `2mm.py` must run first to confirm
> the coordinate mapping works and to produce the CNN dataset. The segmentation
> model reads `labels_dataset.npz` produced by `generateLabels.py`, not
> `dl_dataset_2mm.npz`.

---

## File Details

### 2mm.py — Spatial Dataset Builder

**What it does**

Reads every patient's `shifts_before.mat` file, extracts each spatial scan point's
100-frame depth profile as one training sequence, and saves all sequences to
`dl_dataset_2mm.npz`. Optionally restricts to only the points within 2mm of the
cornea center using X/Y coordinates from `combined_data.xlsx`.

**Why the 2mm filter exists**

The PI's published paper (Zhang et al., AJO 2023) established the central 2mm
region as the clinically relevant zone for keratoconus diagnosis. A radial
separability analysis on this dataset confirmed that points within 2mm have a
Cohen's d of 2.31 (Controls vs SKC), while points beyond 3mm still show strong
separation (Cohen's d = 1.78). The spatial filter is a configurable option so
both the restricted and unrestricted datasets can be compared directly.

**Key configuration**

```python
DATA_DIR    = '/Users/minhnguyen/deep-learning/All Brillouin Point Data'
COORD_FILE  = '/Users/minhnguyen/deep-learning/combined_data.xlsx'
OUTPUT_NPZ  = '/Users/minhnguyen/deep-learning/dl_dataset_2mm.npz'
RADIUS_MM   = 2.0    # set to a large value (e.g. 99) to keep all points
```

**Output**

`dl_dataset_2mm.npz` containing:
- `X`: (n, 100, 1) — depth profile sequences
- `y`: (n,) — binary labels (0=Controls, 1=SKC)
- `groups`: (n,) — patient ID per sequence, for patient-level splitting

---

### generateLabels.py — Binary Cornea Label Generator

**What it does**

For every depth profile in every patient's `.mat` file, determines which of the
100 depth frames correspond to cornea tissue (1) versus aqueous humor or air (0).
These binary label arrays are the ground truth that the segmentation model trains on.

**Why this is needed**

The segmentation model learns to detect the corneal plateau region by training on
examples where the correct answer is already known. Without per-frame labels there
is nothing to train against. This script automates the labeling that was previously
done manually (by Hongyuan in the reference paper).

**How the labeling works**

Each 100-frame depth profile shows a characteristic S-curve shape. The cornea
appears as a low-stiffness plateau at the beginning of the scan (before the
transition to high-stiffness aqueous). The labeling algorithm:

1. Detects the air artifact at the start of each scan using raw-value noise
   (detrended rolling standard deviation) and sets the cornea start frame
2. Applies Otsu's method to find the natural high/low split between cornea and
   aqueous across the full 100-frame signal
3. Labels frames below the Otsu threshold (the low-stiffness cornea region) as
   cornea (1) within the detected cornea window
4. Keeps only the largest contiguous block of labeled frames to eliminate noise
   at the edges

**FSR conversion**

The `.mat` files store plotted Brillouin shift values, not true Brillouin shifts.
The conversion is `true_shift = (FSR - plotted) / 2`, where
`FSR = 2 × 5.07790995626263 + water_before = 15.2337 GHz`. This transform is
decreasing, meaning a lower plotted value maps to a higher true shift. The cornea
(true shift ~5.7 GHz) appears as the LOW plotted region. All GHz values shown
in the plots and summary are in the plotted scale; apply the FSR conversion before
reporting true Brillouin shift values.

**Quality control**

Two QC checks flag sequences before training:

| Check | Condition | Meaning |
|---|---|---|
| Value QC | Labeled cornea mean > 4.0 GHz | Label locked onto aqueous instead of cornea |
| Flatness QC | Mean slope in labeled region > 4× dataset median | Label sits on the transition slope, not the flat plateau |

Flagged sequences are excluded from training. In practice ~53 of 1,081 sequences
(4.9%) are excluded.

**Key configuration**

```python
DATA_DIR       = '/Users/minhnguyen/deep-learning/All Brillouin Point Data'
COORD_FILE     = '/Users/minhnguyen/deep-learning/combined_data.xlsx'
OUTPUT_NPZ     = '/Users/minhnguyen/deep-learning/labels_dataset.npz'
PLOT_DIR       = '/Users/minhnguyen/deep-learning/label_verification_plots'
N_EXAMPLE_PLOTS = 6      # verification plots to generate for manual review
FLAT_FRAC       = 0.25   # slope threshold for flatness QC
EXCLUDE_FLAGGED = True   # set False to keep all sequences for inspection
```

**Output**

`labels_dataset.npz` containing:
- `X`: (n, 100, 1) — raw depth profile sequences
- `Y_seg`: (n, 100) — binary cornea frame labels
- `Y_cls`: (n,) — patient-level diagnosis (0=Controls, 1=SKC)
- `groups`: (n,) — patient ID per sequence
- `qc_flags`: (n,) — 1 if flagged, 0 if kept
- `plateau_means`: (n,) — mean plotted shift of labeled cornea frames

6 PNG verification plots saved to `PLOT_DIR`. **Review these with your mentor before
training the segmentation model.** Each plot shows the depth profile, the Otsu
threshold, the detected cornea start frame (magenta line), and the labeled cornea
frames (yellow dots).

---

### segmentation.py — 1D U-Net Segmentation and Stiffness Measurement

**What it does**

Trains a 1D U-Net to predict the binary cornea label array for new, unseen depth
profiles. After training, takes the mean of the predicted cornea frames as a
per-point stiffness estimate, averages those estimates per patient, and produces a
Controls-vs-SKC box plot along with all five metrics from Zhang et al. Table 4.

**Architecture**

The model follows the encoder-decoder structure of the TensorFlow segmentation
tutorial (U-Net style) adapted from 2D images to 1D signals:

```
Input: (100, 1)
  Conv1D(32, kernel=5, relu) → Conv1D(32, kernel=5, relu) → MaxPooling1D(2)  # 100→50
  Conv1D(64, kernel=5, relu) → Conv1D(64, kernel=5, relu) → MaxPooling1D(2)  # 50→25
  Conv1D(128, kernel=5, relu) → Conv1D(128, kernel=5, relu) → Dropout(0.3)   # bottleneck
  UpSampling1D(2) + Concatenate(skip from encoder) → Conv1D(64)×2            # 25→50
  UpSampling1D(2) + Concatenate(skip from encoder) → Conv1D(32)×2            # 50→100
  Conv1D(1, kernel=1, sigmoid)   # per-frame cornea probability
```

Input length 100 halves cleanly to 50 then 25 (both even), so the skip connections
concatenate without any cropping. This is the key dimensional constraint that makes
a 2-level U-Net the right choice for 100-frame sequences.

**Training**

- Patient-level GroupShuffleSplit: ~24 train patients, ~6 test patients per split
- StandardScaler fitted on training data only (prevents normalization leakage)
- EarlyStopping with patience=8, monitoring validation loss
- 5 repeated splits averaged for a stable result (a single 6-patient test split
  is too noisy to trust — one unlucky draw can appear to show no separation between
  groups even when the overall dataset does)
- Loss: binary cross-entropy on per-frame predictions

**Downstream metrics**

After segmentation, the pipeline computes the five spatial metrics from
Zhang et al. AJO 2023 (Table 4) per patient, in true Brillouin shift (GHz):

| Metric | Description | Paper AUC (Controls vs KC) |
|---|---|---|
| Mean | Average over predicted cornea frames | 0.96 |
| Max | Highest (stiffest) predicted frame | — |
| Min | Lowest (softest) predicted frame | 1.00 |
| Min_p10 | 10th percentile (noise-robust Min) | n/a |
| SSD | Spatial standard deviation | 0.82 |
| Max-Min | Stiffness range | 0.87 |

> **Note on Min:** The paper's Min achieved AUC 1.00 because their manual
> measurements had very low per-point noise (spatial SD 0.012 GHz). This
> automated pipeline has higher per-point noise (spatial SD ~0.030 GHz), so
> the absolute minimum chases noise rather than the true softest point. Min_p10
> (10th percentile) is provided as a noise-robust alternative.

**Key configuration**

```python
DATASET_NPZ  = '/Users/minhnguyen/deep-learning/labels_dataset.npz'
PLOT_DIR     = '/Users/minhnguyen/deep-learning/segmentation_plots'
RESTRICT_2MM = False    # set True to train on central 2mm points only
COORD_FILE   = '/Users/minhnguyen/deep-learning/combined_data.xlsx'
N_SPLITS     = 5
EPOCHS       = 40
```

**Results**

Averaged over 5 patient splits:
- Per-frame segmentation accuracy: **94.8%**
- Dice coefficient: **0.913** (1.0 = perfect overlap)
- Per-patient cornea mean: Controls 3.705 GHz vs SKC 3.765 GHz (plotted scale)
- In true Brillouin shift: Controls ~5.764 GHz vs SKC ~5.733 GHz
- p-value: < 0.0001

The 2mm spatial restriction was also tested. It produced weaker separation
(Cohen's d 1.18 vs 1.43 for all points), likely because restricting to 444 sequences
from 1,028 loses statistical power even though the central region carries stronger
per-point signal. Both results are reported for the scientific comparison.

**Output files**

- `cornea_mean_boxplot.png` — two-panel box plot (point level and patient level)
- `paper_metrics_boxplot.png` — six-panel plot, one per metric, with Zhang et al.
  reference values overlaid as dashed lines
- `per_patient_metrics.csv` — one row per patient per split, all six metrics,
  ready to receive Hongyuan's manual values as an additional column for direct
  comparison

---

### cnn3.py — 1D CNN Classifier (for comparison)

**What it does**

An alternative deep learning approach that classifies depth profiles directly
without segmentation labels. Three versions were developed:

- **v1** (`cnn.py`): single 80/20 split, fixed 0.5 threshold, Dense(64). Validation
  accuracy stalled at 44.7% — same failure mode as the LSTM.
- **v2** (`cnn2.py`): added 10 repeated splits, threshold tuning on validation set,
  reduced Dense to 32. Achieved 65% patient accuracy but unstable (7/10 splits
  learned real separation, 2 were noise, 1 was inverted).
- **v3** (`cnn3.py`): added temporal jitter augmentation (+/- 5 frames per training
  sequence) to force the model to learn the transition zone shape rather than
  memorizing frame positions. Result: accuracy dropped slightly (58.3%) and standard
  deviation increased, confirming the bottleneck is per-point SNR (1.48), not
  architecture. Included as the final version for the scientific record.

**Why the CNN underperforms the ML pipeline**

The between-class signal at the individual sequence level (0.036 GHz) is smaller
than the within-class noise (0.024 GHz), giving a per-point signal-to-noise ratio
of 1.48. The ML pipeline achieves 90–97% by averaging ~36 sequences per patient,
which cuts noise by √36 ≈ 6×, raising the SNR to 8.91. No CNN architecture can
overcome an SNR below 1 on individual sequences without substantially more data
or a different input representation.

**Key configuration**

```python
DATASET_NPZ = '/Users/minhnguyen/deep-learning/dl_dataset_2mm.npz'
EPOCHS      = 50
N_SPLITS    = 10
MAX_SHIFT   = 5    # temporal jitter magnitude in frames
```

---

## Dependencies

```
numpy
pandas
scipy
scikit-learn
tensorflow
matplotlib
openpyxl
```

Install with:
```bash
/Users/minhnguyen/dl-env/bin/pip install numpy pandas scipy scikit-learn tensorflow matplotlib openpyxl
```

---

## Data Files Required

| File | Description | Produced by |
|---|---|---|
| `shifts_before.mat` | Per-patient Brillouin shifts, 100×N array | Lab instrument |
| `combined_data.xlsx` | Patient, Plateau, X(mm), Y(mm), Diagnosis | plateauXY.py |
| `dl_dataset_2mm.npz` | CNN training sequences | 2mm.py |
| `labels_dataset.npz` | Segmentation training sequences + labels | generateLabels.py |
| `fb_water.mat` | Water calibration reference (fB_water = 5.0779 GHz) | Lab instrument |

---

## Reference

Zhang H, Asroui L, Tarib I, Dupps WJ, Scarcelli G, Randleman JB.
*Motion-Tracking Brillouin Microscopy Evaluation of Normal, Keratoconic, and
Post–Laser Vision Correction Corneas.*
Am J Ophthalmol. 2023;254:128–140.

Table 4 of this paper provides the benchmark values (Controls vs KC mean, max, min,
SSD, max-min Brillouin shift) against which the automated segmentation pipeline's
output is compared.

---

*Developed as part of a Brillouin microscopy internship at the University of
Maryland Fischell Department of Bioengineering under the supervision of Ron and
Giuliano Scarcelli. Dataset: 30 patients (15 Controls, 15 SKC).*
