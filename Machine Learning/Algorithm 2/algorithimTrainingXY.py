"""
Algorithm 2 - ML Training with Spatial Coordinates (algorithimTrainingXY.py)
=============================================================================

WHAT THIS SCRIPT DOES
----------------------
Extends Algorithm 1 by training classifiers on THREE features per scan point
instead of one aggregated mean per patient:
  1. Plateau  — Brillouin frequency shift (stiffness) at this spatial location
  2. X (mm)   — horizontal position of this scan point on the cornea surface
  3. Y (mm)   — vertical position of this scan point on the cornea surface

HOW THIS DIFFERS FROM ALGORITHM 1
-----------------------------------
Algorithm 1 averaged all of a patient's scan points into a single number
and classified patients (N=30). Algorithm 2 classifies individual scan
POINTS (N~1,081) while still treating the PATIENT as the fundamental unit
for train/test splitting. This allows the model to potentially learn that
central points (small X/Y) matter more for KC diagnosis than peripheral
points — something Algorithm 1's patient-averaged approach cannot capture.

The tradeoff: individual points are noisier than patient averages (SNR ~1.48
vs ~8.91), so you should expect lower accuracy than Algorithm 1 in the
Leave-One-Patient-Out evaluation. The point-level classification accuracy
is also not directly comparable to Algorithm 1's patient-level accuracy
without a per-patient aggregation step (e.g. majority vote).

VALIDATION APPROACH
--------------------
Two evaluations are run:
  1. Single Split (80/20): trains on 80% of PATIENTS and tests on 20%.
     A fast diagnostic. The train-vs-validation gap reveals overfitting.
     NOT the primary result — treat as exploratory only.
  2. Leave-One-Patient-Out (LOGO): holds out one complete patient at a time,
     trains on everyone else, repeats for all patients. The primary result.
     This is the most honest estimate of real-world performance at this N.

INPUT  : combined_data.xlsx   (from plateauXY.py)
OUTPUT : classifier comparison printed to terminal
"""

import pandas as pd
from sklearn.ensemble import (BaggingClassifier, ExtraTreesClassifier,
                               GradientBoostingClassifier, RandomForestClassifier)
from sklearn.model_selection import (GroupShuffleSplit, LeaveOneGroupOut,
                                      cross_val_score)
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


# ── DATA LOADING & PREPROCESSING ──────────────────────────────────────────
# Load the combined per-point dataset produced by plateauXY.py.
# This file has one row per spatial scan point across all 30 patients,
# with columns: Patient, Plateau, X (mm), Y (mm), Diagnosis.
COMBINED_FILE = '/Users/minhnguyen/try-scipy/combined_data.xlsx'
dataset = pd.read_excel(COMBINED_FILE)

# Drop any rows where Plateau, X, or Y is missing (NaN). With three features
# instead of one, we need all three to be valid for a data point to be usable.
# Missing values can occur from blank rows at the bottom of an Excel sheet.
dataset = dataset.dropna(subset=['Plateau', 'X (mm)', 'Y (mm)'])
print(f"Total measurements after removing missing values: {len(dataset)}")

# ── FEATURES AND TARGET ────────────────────────────────────────────────────
# X: the feature matrix — each row is one scan point, three columns.
#    Plateau is in GHz (~3.7–4.7 plotted, ~5.6–5.8 true Brillouin shift).
#    X (mm) and Y (mm) are the corneal surface coordinates of this point.
# y: the diagnosis label ('Controls' or 'SKC') for the patient this point
#    belongs to. Every point from the same patient gets the same label.
# groups: the patient ID for each row. This is critical for preventing
#    data leakage — all points from the same patient must stay together
#    on the same side of any train/test split.
X      = dataset[['Plateau', 'X (mm)', 'Y (mm)']]
y      = dataset['Diagnosis']
groups = dataset['Patient']


# ── TRAIN/TEST SPLIT (PATIENT-LEVEL) ──────────────────────────────────────
# WHY GroupShuffleSplit instead of a regular split:
#
# Each patient has ~30–44 scan points. If we split randomly by row, the same
# patient's points can appear in BOTH the training set and the validation set.
# The classifier then evaluates itself partly on data it effectively trained on
# (since it saw the same patient's pattern), inflating validation accuracy.
# This is called patient-level data leakage and is a common trap in medical ML.
#
# GroupShuffleSplit splits by patient GROUP, not by row. With groups=Patient,
# every patient is guaranteed to be either 100% in training OR 100% in test.
# train_size=0.8 means approximately 24 patients train, 6 patients test.
# random_state=3 ensures the same split is produced every time for reproducibility.
gss = GroupShuffleSplit(n_splits=1, train_size=0.8, random_state=3)
train_idx, test_idx = next(gss.split(X, y, groups))

X_train,      X_validation = X.iloc[train_idx],      X.iloc[test_idx]
Y_train,      Y_validation = y.iloc[train_idx],      y.iloc[test_idx]

# Show how many Controls and SKC measurements are in each split.
# Ideally both splits should have a mix of both classes.
print("── TRAINING DIAGNOSIS BREAKDOWN ──")
print(Y_train.value_counts())
print("\n── VALIDATION DIAGNOSIS BREAKDOWN ──")
print(Y_validation.value_counts())
print("───────────────────────────────────\n")


# ── MODEL SELECTION ────────────────────────────────────────────────────────
# 11 classifiers are tested. With three features rather than one, we can
# explore models that benefit from multi-dimensional decision boundaries
# (SVM with RBF kernel, ensemble methods, LDA, GB).
#
# Regularization philosophy: tighter than sklearn defaults throughout.
# Algorithm 2 trains on ~870 scan points from ~24 patients (in the LOGO
# setting, ~24 patients' points = ~870 rows with many correlated rows from
# the same patient). Overfitting is a real risk, especially for flexible
# models like trees and ensembles.
#
# class_weight='balanced': ensures both classes contribute equally to the
# loss even if one has slightly more scan points than the other.
#
# StandardScaler in a pipeline: Plateau is in GHz (~3.7–4.7 range),
# X and Y are in mm (~-4 to +4 range). These are on different numeric scales.
# Distance-based models (KNN, SVM) and regularization-based models (LR, LDA,
# Ridge) are sensitive to feature scale and will underperform without scaling.
# Wrapping them in make_pipeline ensures scaling is fit ONLY on training data
# and then applied to validation, preventing scale leakage.

models = []

# Logistic Regression: linear boundary in the 3D feature space.
# C=1.0 is slightly less regularized than Algorithm 1's C=0.5, since with
# 3 features there's more signal to learn without overfitting as badly.
models.append(('LR', make_pipeline(
    StandardScaler(),
    LogisticRegression(solver='liblinear', C=1.0, class_weight='balanced')
)))

# Linear Discriminant Analysis: assumes both classes have the same covariance
# structure. Finds the linear combination of features that best separates the
# classes. No hyperparameters to tune; a useful baseline for multi-feature data.
models.append(('LDA', LinearDiscriminantAnalysis()))

# K-Nearest Neighbors: classifies each point by looking at the 3 nearest
# training points in the 3D (Plateau, X, Y) feature space. n_neighbors=3
# (vs 5 in Algorithm 1) because the per-point dataset is larger.
# NOTE: KNN is unscaled here. Plateau (GHz) and X/Y (mm) have different
# magnitudes, which will bias KNN toward the feature with larger raw values.
models.append(('KNN', KNeighborsClassifier(n_neighbors=3)))

# Decision Tree (CART): max_depth=3 allows slightly deeper trees than
# Algorithm 1 (depth=2) since we have more features to split on, but still
# shallow enough to prevent the tree from memorizing training patients.
# min_samples_leaf=2 prevents leaves from fitting to just 1–2 data points.
models.append(('CART', DecisionTreeClassifier(
    class_weight='balanced', max_depth=3, min_samples_leaf=2
)))

# Naive Bayes: assumes each feature independently follows a Gaussian
# distribution within each class. Simple, fast, and a useful sanity check.
models.append(('NB', GaussianNB()))

# Support Vector Machine with RBF (radial basis function) kernel.
# gamma='scale' sets gamma = 1 / (n_features * X.var()), adapting to the
# feature variance. The RBF kernel can capture non-linear separating
# boundaries, which may be useful if the central vs peripheral spatial
# pattern is non-linear. C=1.0 balances margin width against training errors.
models.append(('SVM', make_pipeline(
    StandardScaler(),
    SVC(gamma='scale', class_weight='balanced', C=1.0)
)))

# Random Forest: 100 trees, each trained on a random subset of patients and
# a random subset of features. max_depth=4 and min_samples_leaf=2 prevent
# individual trees from overfitting to specific patient signatures.
# n_estimators=100 is larger than Algorithm 1 (50) because the per-point
# dataset is larger and can support a bigger ensemble.
models.append(('RF', RandomForestClassifier(
    class_weight='balanced', random_state=1,
    max_depth=4, min_samples_leaf=2, n_estimators=100
)))

# Gradient Boosting: trains trees sequentially, each one correcting the
# errors of the previous. More expressive than Random Forest but also more
# prone to overfitting. max_depth=3 keeps individual trees shallow.
# No class_weight support in sklearn's GradientBoostingClassifier.
models.append(('GB', GradientBoostingClassifier(random_state=1, max_depth=3)))

# Extra Trees: similar to Random Forest but uses completely random split
# thresholds instead of optimizing them. Typically faster and sometimes
# more regularized than RF.
models.append(('ET', ExtraTreesClassifier(
    class_weight='balanced', random_state=1,
    max_depth=4, min_samples_leaf=2, n_estimators=100
)))

# Ridge Classifier: linear model with L2 regularization on the coefficients.
# alpha=1.0 is the regularization strength (higher = more regularized).
# NOTE: Ridge is unscaled here, unlike LR and SVM. Since Plateau, X, and Y
# are on different scales, Ridge may not learn the optimal coefficient weights.
models.append(('Ridge', RidgeClassifier(class_weight='balanced', alpha=1.0)))

# Bagging (Bootstrap Aggregating): trains 25 independent base classifiers on
# random 80% subsets of the training data (max_samples=0.8). The default
# base estimator is a Decision Tree. Reduces variance through averaging,
# similar to Random Forest but without the feature randomization.
models.append(('Bag', BaggingClassifier(
    random_state=1, n_estimators=25, max_samples=0.8
)))


# ── SINGLE SPLIT EVALUATION ────────────────────────────────────────────────
# This is a quick diagnostic using the 80/20 patient-level split defined above.
# It shows whether a model is overfitting to the training patients.
#
# Train-vs-validation gap interpretation:
#   Small gap   (< 10%): model generalizes reasonably on this split
#   Moderate    (10–20%): some overfitting; treat result with caution
#   Large gap   (> 20%): significant overfitting, especially dangerous at N=30
#   Negative gap: validation is accidentally easier than training — can happen
#                 when the random split gives a lucky validation set; not a
#                 sign of super-generalization
#
# This single split is NOT the primary result. With only 6 test patients,
# one wrong prediction is worth 16.7% and the result is highly dependent on
# which patients randomly landed in the test set.
print("── SINGLE SPLIT RESULTS (Train vs Validation) ──")
for name, model in models:
    try:
        model.fit(X_train, Y_train)
        train_score = accuracy_score(Y_train, model.predict(X_train))
        val_score   = accuracy_score(Y_validation, model.predict(X_validation))
        gap = train_score - val_score
        print(f"{name}:  Train: {train_score*100:.1f}%  |  "
              f"Validation: {val_score*100:.1f}%  |  Gap: {gap*100:.1f}%")
    except Exception as e:
        print(f"{name}: ERROR — {e}")


# ── LEAVE-ONE-PATIENT-OUT CROSS-VALIDATION (PRIMARY RESULT) ───────────────
# How LOGO works:
#   For each unique patient P in the dataset:
#     - Hold out ALL of patient P's scan points as the test set
#     - Train on every other patient's scan points
#     - Evaluate the model on patient P's held-out points
#     - Record the accuracy for this fold
#   Report the mean accuracy across all patient folds.
#
# Why LOGO is the correct method here:
#   - With 30 patients, the 80/20 single split above tests on only ~6 patients.
#     Accuracy can swing wildly based on which 6 patients are chosen.
#   - LOGO tests every patient exactly once and uses the maximum amount of
#     data for training at each fold (~29 patients' worth of scan points).
#   - By passing groups=groups (patient IDs), sklearn ensures that ALL points
#     from the held-out patient are removed from training, eliminating leakage.
#
# Important note on what LOGO accuracy means here:
#   The reported number is the fraction of INDIVIDUAL SCAN POINTS correctly
#   classified across all patients — not the fraction of patients correctly
#   diagnosed. One patient's LOGO accuracy depends on whether the model gets
#   each of their ~36 scan points right. This is a harder problem than
#   Algorithm 1's patient-averaged classification.
logo = LeaveOneGroupOut()

print("\n── LEAVE-ONE-PATIENT-OUT RESULTS ──")
for name, model in models:
    try:
        # cross_val_score handles the LOGO splitting automatically.
        # It returns one accuracy score per held-out patient, then we average.
        scores = cross_val_score(model, X, y,
                                 cv=logo, groups=groups, scoring='accuracy')
        print(f"{name}: {scores.mean():.4f} ({scores.mean()*100:.1f}% avg accuracy)")
    except Exception as e:
        print(f"{name}: ERROR — {e}")