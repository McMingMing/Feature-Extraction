# Brillouin Microscopy Corneal Classification
### University of Maryland — Fischell Institute for Biomedical Devices

Automated classification of **Subclinical Keratoconus (SKC)** vs healthy **Controls**
using Brillouin microscopy measurements of corneal stiffness. This project implements
and compares classical machine learning and deep learning approaches on a dataset of
30 patients (15 Controls, 15 SKC).

---

## Background

Keratoconus (KC) is a progressive corneal disease where the cornea thins and bulges
outward, causing visual impairment. **Subclinical Keratoconus (SKC)** is the early,
clinically undetectable stage where structural damage has begun but symptoms have not
yet appeared. Detecting SKC before refractive surgery is critical — operating on an
undetected KC eye accelerates the disease.

Brillouin microscopy measures corneal stiffness non-invasively by analyzing how laser
light scatters off the tissue. Stiffer tissue produces a higher Brillouin frequency
shift. Each patient's cornea is scanned at ~30–44 spatial locations within an 8mm
circle. At each location, a 100-frame depth profile captures how stiffness varies
through the cornea's thickness. This project uses those measurements to automatically
distinguish SKC from healthy Controls.

**Reference:** Zhang H et al., *Motion-Tracking Brillouin Microscopy Evaluation of
Normal, Keratoconic, and Post–Laser Vision Correction Corneas.*
Am J Ophthalmol. 2023;254:128–140. (Hongyuan's manual pipeline is the benchmark
this project automates.)

---

## Repository Structure

```
Algorithm 1 — ML, patient-averaged
├── dataAutomation.py          Data prep: reads per-patient Excel files → combined_plateau.xlsx
└── algorithimTraining.py      ML: 8 classifiers on mean plateau per patient

Algorithm 2 — ML, per-point with spatial coordinates
├── plateauXY.py               Data prep: reads Excel + X/Y coords → combined_data.xlsx
└── algorithimTrainingXY.py    ML: 11 classifiers on (Plateau, X, Y) per point

.mat-based ML pipeline (spatial feature engineering)
├── masterdataSet.py           Feature engineering: 5 spatial features → ML_Master_Dataset.xlsx
└── classifiers.py             ML: 8 classifiers on engineered features with CV

Deep Learning — LSTM
├── buildDataSet.py            DL data: extracts depth profiles → dl_dataset.npz
└── runLTSM.py                 LSTM classifier on 100-frame sequences

Deep Learning — CNN and Segmentation
├── 2mm.py                     DL data: extracts depth profiles (optional 2mm filter) → dl_dataset_2mm.npz
├── generateLabels.py          Generates binary cornea labels → labels_dataset.npz
├── segmentation.py            1D U-Net segmentation + stiffness metrics + box plots
└── cnn3.py                    1D CNN classifier (final version, v1→v2→v3 progression)
```

---

## Pipelines at a Glance

| Pipeline | Input | Feature | N samples | Best accuracy |
|---|---|---|---|---|
| Algorithm 1 | Excel plateau files | Mean plateau per patient | 30 patients | 93–97% LOOCV |
| Algorithm 2 | Excel plateau + XY | Plateau, X, Y per point | ~1,081 points | 84.3% LOGO |
| .mat ML | shifts_before.mat + combined_data.xlsx | 5 spatial features per patient | 30 patients | 93–97% LOOCV |
| LSTM | shifts_before.mat | 100-frame depth profile | ~1,081 sequences | ~33% patient (failed) |
| CNN | shifts_before.mat (2mm) | 100-frame depth profile | ~449 sequences | 58–65% patient |
| Segmentation | shifts_before.mat | Predicted cornea frames | ~1,028 sequences | 94.8% Dice, p<0.0001 |

---

## Quick Start

### Algorithm 1

```bash
python dataAutomation.py
python algorithimTraining.py
```

### Algorithm 2

```bash
python plateauXY.py
python algorithimTrainingXY.py
```

### .mat-based ML

```bash
python masterdataSet.py
python classifiers.py
```

### Deep Learning — LSTM

```bash
python buildDataSet.py
python runLTSM.py
```

### Deep Learning — Segmentation (recommended over LSTM)

```bash
python 2mm.py               # build depth-profile dataset
python generateLabels.py    # generate cornea frame labels
python segmentation.py      # train U-Net, produce metrics and box plots
python cnn3.py              # optional: CNN comparison
```

---

## Data Files

| File | Description | Produced by |
|---|---|---|
| Per-patient `.xlsx` files | One per patient, plateau + X/Y coordinates | Lab instrument |
| `SKC_names.xlsx` | Diagnosis key: Controls and SKC patient ID lists | Manual |
| `fb_water.mat` | Water calibration: fB_water = 5.0779 GHz | Lab instrument |
| `shifts_before.mat` | Per-patient Lorentzian-fitted Brillouin shifts (100×N) | Lab pipeline |
| `combined_plateau.xlsx` | Patient, Plateau, Diagnosis (one row per scan point) | dataAutomation.py |
| `combined_data.xlsx` | Patient, Plateau, X(mm), Y(mm), Diagnosis | plateauXY.py |
| `ML_Master_Dataset.xlsx` | One row per patient with 5 spatial features | masterdataSet.py |
| `dl_dataset.npz` | All depth profiles for LSTM | buildDataSet.py |
| `dl_dataset_2mm.npz` | Central 2mm depth profiles for CNN | 2mm.py |
| `labels_dataset.npz` | Depth profiles + binary cornea labels for U-Net | generateLabels.py |

---

## Pipeline Details

### Algorithm 1 — Classical ML, Patient Averaged

The simplest approach. All of a patient's ~36 scan points are averaged into one
Brillouin plateau value, then fed into eight classifiers. One feature per patient,
30 training samples total.

**Why it works:** averaging ~36 noisy per-point measurements cuts the noise by √36
≈ 6×, raising the signal-to-noise ratio from 1.48 (per point) to 8.91 (per
patient). At SNR 8.91, even a simple threshold on the mean cleanly separates the
two groups.

**Validation:** Leave-One-Patient-Out CV. Trains on 29 patients, tests on 1,
repeated 30 times. Most classifiers reach **90–97% accuracy** (27–29/30 correct).
95% confidence intervals are wide (~±10%) at N=30 and should always be reported
alongside the headline number.

→ See `README_algorithm1.md` for full details.

---

### Algorithm 2 — Classical ML, Per-Point with Spatial Coordinates

Extends Algorithm 1 by keeping each scan point's X/Y position as additional
features. Classifies individual points rather than patient averages.

**Three features:** Plateau, X (mm), Y (mm). The model can learn that central
points (near X=0, Y=0) carry more diagnostic information than peripheral ones.

**Key finding from this pipeline:** a radial separability analysis showed that
every radius bin (center to periphery) has large, statistically significant
Controls vs SKC separation (Cohen's d 1.24–2.31). The periphery is not noise.
This finding informed the spatial filtering strategy in the deep learning pipeline.

**Best result:** SVM at **84.3% point-level LOGO accuracy**. Lower than Algorithm 1
because per-point SNR is 1.48 versus 8.91 after patient averaging. Not directly
comparable to Algorithm 1 without majority voting.

→ See `README_algorithm2.md` for full details.

---

### .mat-based ML Pipeline — Spatial Feature Engineering

Uses the Lorentzian-fitted `.mat` files (post-processed stiffness values) with
the X/Y coordinate mapping from `combined_data.xlsx` to engineer five spatial
features per patient. More principled than Algorithm 1's simple average because
it explicitly restricts to the clinically relevant central 2mm region and captures
the spatial distribution of stiffness.

**Five features:**

| Feature | Description |
|---|---|
| `Mean_Plateau_2mm` | Mean stiffness within 2mm of center (dominant discriminator) |
| `Mean_Plateau_All` | Mean stiffness across all points |
| `Std_Plateau_2mm` | Stiffness uniformity within 2mm (low = healthy, high = KC) |
| `Center_Periphery_Gradient` | Mean(inner < 1mm) − Mean(mid 1–2mm) |
| `Min_Plateau_2mm` | Softest localized point within 2mm |

**Key finding:** `Mean_Plateau_2mm` alone as a single feature matches the
five-feature model (93.3% vs 96.7% LOOCV). The spatial features add little beyond
the central mean at N=30.

→ See `README_mat_pipeline.md` for full details.

---

### Deep Learning — LSTM

A stacked 3-layer LSTM trained on individual 100-frame depth profiles, following
the architecture from Ron's template. Each training sample is one spatial scan
point's depth profile (one column of `shifts_before.mat`).

**Result:** validation accuracy stalled at 44.7% for all 50 epochs. Patient-level
accuracy 33.3% — equivalent to a coin flip. **The LSTM failed.**

**Diagnosis:** the between-class signal at the individual sequence level (0.036 GHz)
is smaller than the within-class noise (0.024 GHz), giving a per-point
signal-to-noise ratio of 1.48. No LSTM architecture can overcome SNR below 1 at
this sample size without substantially more data or a different input representation.
Three fixes were tried (wider layers, more epochs, class weighting) — none moved
validation accuracy.

→ See `README_mat_pipeline.md` for architecture details and full failure analysis.

---

### Deep Learning — CNN (cnn3.py)

A 1D CNN alternative to the LSTM. Three versions were developed:

- **v1:** single split, fixed threshold, Dense(64). Validation stuck at 44.7%.
- **v2:** 10 repeated splits, threshold tuning, Dense(32). 65% patient accuracy
  but unstable (7/10 splits learned real separation, 2 noise, 1 inverted).
- **v3:** added temporal jitter augmentation to force the model to learn the
  S-curve transition shape rather than memorizing frame positions. Result:
  accuracy dropped to 58.3%, confirming the bottleneck is per-point SNR, not
  architecture.

**Conclusion:** three architectures (LSTM, CNN, augmented CNN) all converged to
the same 58–65% ceiling. This is a property of the data, not a tuning problem.

→ See `README_segmentation_pipeline.md` for full details.

---

### Deep Learning — 1D U-Net Segmentation (recommended)

The most successful deep learning approach. Instead of classifying each depth
profile directly, the U-Net learns to identify which frames within a profile
correspond to corneal tissue. The mean stiffness of the predicted cornea frames
is then computed per patient for the final Controls vs SKC comparison.

**Why this works while the CNN does not:** segmentation is a per-frame detection
task rather than a whole-sequence classification task. The model only needs to
find where the low-stiffness plateau sits within each profile, which is a more
locally defined and learnable signal than classifying the whole sequence.

**Architecture:** 1D U-Net with 2 pooling levels. Input length 100 halves cleanly
to 50 then 25 at the bottleneck, so skip connections concatenate without cropping.
Encoder compresses the sequence; decoder reconstructs a per-frame binary mask.

```
Input (100,1) → Conv×2 → Pool → Conv×2 → Pool → Bottleneck
              ← Upsample + Skip ← Conv×2 ← Upsample + Skip ← Conv×2
Output (100,1): per-frame cornea probability
```

**Label generation (`generateLabels.py`):** binary cornea labels (1=cornea,
0=other) are generated automatically using Otsu thresholding and flatness-based
QC. The FSR conversion `true_shift = (FSR − plotted) / 2` (FSR = 15.2337 GHz)
converts plotted values to true Brillouin shifts. Cornea appears at the LOW plotted
value (~3.75 GHz plotted = ~5.74 GHz true). ~53 of 1,081 sequences (4.9%) are
excluded by QC before training.

**Downstream metrics** computed per patient across the predicted cornea frames,
matched against Zhang et al. AJO 2023 Table 4:

| Metric | Our result (Controls vs SKC) | Paper (Controls vs KC) |
|---|---|---|
| Mean | 5.764 vs 5.733 GHz, p<0.0001 | 5.713 vs 5.679 GHz, p=0.0004 |
| Max | significant, p=0.0026 | p=0.089 |
| Min | significant, p=0.042 | p=0.000002, AUC=1.00 |
| Min_p10 | significant, p=0.0047 | n/a (robust alternative) |
| SSD | significant, p=0.036 | p=0.03 |

> **Note on Min:** the paper's Min achieved AUC 1.00 because their manual
> per-point measurements had very low noise (spatial SD 0.012 GHz). This pipeline's
> automated estimates are noisier (SD ~0.030 GHz), so the absolute minimum chases
> noise. `Min_p10` (10th percentile) is provided as a robust alternative.

**Segmentation quality:** 94.8% per-frame accuracy, 0.913 Dice coefficient,
averaged over 5 repeated patient splits with very low variance.

→ See `README_segmentation_pipeline.md` for full details.

---

## Key Findings Summary

| Finding | Detail |
|---|---|
| Single feature is sufficient | Mean plateau within 2mm alone achieves 93.3% LOOCV accuracy — same as 5 features |
| Per-point SNR is the DL bottleneck | Individual sequences have SNR 1.48; patient averaging raises it to 8.91 |
| Periphery carries real signal | Cohen's d = 1.78 beyond 3mm — spatial restriction to center does not help DL |
| Segmentation reproduces paper direction | Controls stiffer than SKC in all metrics, matching Zhang et al. direction |
| Min metric is noise-limited | Our per-point noise floor (0.030 GHz) exceeds the paper's full KC signal (0.018 GHz) |
| Multi-split evaluation is essential | A single 6-patient test split gave Controls ≈ SKC (noise); 5-split average recovered p<0.0001 |

---

## Dependencies

```bash
pip install numpy pandas scipy scikit-learn tensorflow matplotlib openpyxl
```

| Package | Used by |
|---|---|
| numpy, pandas | All pipelines |
| scikit-learn | All ML and DL pipelines (splitting, scaling, metrics) |
| scipy | generateLabels.py (Gaussian filter, Otsu), segmentation.py (t-test) |
| tensorflow | runLTSM.py, cnn3.py, segmentation.py |
| matplotlib | generateLabels.py, segmentation.py (plots) |
| openpyxl | All pipelines that read/write .xlsx files |

---

## Mentors and Affiliations

Internship project at the **University of Maryland Fischell Department of
Bioengineering** under the supervision of **Ron** and **Giuliano Scarcelli**.

Dataset: 30 patients (15 Controls, 15 SKC), collected using the motion-tracking
Brillouin microscope described in Zhang et al. AJO 2023.

---

## Further Reading

- `README_algorithm1.md` — detailed documentation for the patient-averaged ML pipeline
- `README_algorithm2.md` — detailed documentation for the per-point ML pipeline
- `README_mat_pipeline.md` — detailed documentation for the .mat ML and LSTM pipelines
- `README_segmentation_pipeline.md` — detailed documentation for the CNN and U-Net pipelines
