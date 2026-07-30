"""
Algorithm 1 - ML Training (algorithimTraining.py)
==================================================

WHAT THIS SCRIPT DOES
----------------------
Takes the output of dataAutomation.py (combined_plateau.xlsx — one row per
spatial point across a patient's cornea scan, with columns: Patient, Plateau,
Diagnosis) and runs a machine learning classification pipeline on it to
determine whether a patient has Subclinical Keratoconus (SKC) or is a healthy
Control.

PIPELINE IN ORDER:
  1. Load combined_plateau.xlsx (produced by dataAutomation.py)
  2. Aggregate: collapse the ~30+ per-point plateau readings for each patient
     down to ONE number — the mean Brillouin plateau shift across all their
     spatial scan points
  3. Feed that one number per patient into 8 different ML classifiers
  4. Evaluate every classifier with Leave-One-Patient-Out Cross-Validation
     (the most honest evaluation method at N=30) and report accuracy + 95% CI

WHY AVERAGE FIRST?
  Each patient has ~30–44 spatial measurement points across their cornea.
  Each point gives one plateau value. Averaging them gives a single stable
  estimate of that patient's overall corneal stiffness. This reduces noise
  and lets every classifier work at the patient level (30 samples) rather
  than the point level (1,081 samples with correlated data from the same
  patient mixed across train and test — a form of data leakage).

WHY LEAVE-ONE-PATIENT-OUT CV?
  With only 30 patients, standard random 80/20 splits are unreliable. LOOCV
  trains on 29 patients and tests on the 1 held out, repeated 30 times so
  every patient gets tested exactly once. The fraction correct across all 30
  rounds is the accuracy estimate. One wrong prediction = 3.3 percentage
  points, so always read the 95% confidence interval alongside the headline
  number.

INPUT  : combined_plateau.xlsx  (output of dataAutomation.py)
OUTPUT : Algorithm1_Master_Dataset.xlsx  (one row per patient: Patient,
         Mean_Plateau, Diagnosis) and classifier results printed to terminal.
"""

import pandas as pd
import numpy as np
import warnings
from sklearn.model_selection import LeaveOneOut, cross_val_score
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# Suppress sklearn convergence and other minor warnings that clutter output
warnings.filterwarnings("ignore")

# ── CONFIGURATION ─────────────────────────────────────────────────────────
INPUT_FILE  = 'combined_plateau.xlsx'           # produced by dataAutomation.py
OUTPUT_FILE = 'Algorithm1_Master_Dataset.xlsx'  # one row per patient, for inspection

# The single feature fed into every classifier.
# This is the per-patient mean of all their spatial Brillouin plateau readings.
# The mentor's original instruction: "take the average of the plateau data and
# run the ML code." We keep it at one feature for simplicity and interpretability.
# A one-feature model that works is more trustworthy at N=30 than a
# five-feature model that might be overfitting.
FEATURES = ['Mean_Plateau']
# ──────────────────────────────────────────────────────────────────────────


def build_patient_table(df):
    """
    Collapse all per-point rows for each patient into a single row containing
    just the mean Brillouin plateau value and that patient's diagnosis label.

    Why .first() for Diagnosis: every row for the same patient has the same
    diagnosis label, so we just take the first occurrence. groupby().agg()
    requires us to specify how to aggregate every column explicitly.

    Args:
        df: DataFrame with columns Patient, Plateau, Diagnosis
            (one row per spatial measurement point)

    Returns:
        DataFrame with one row per patient, columns:
        Patient | Mean_Plateau | Diagnosis
    """
    agg = df.groupby('Patient').agg(
        Mean_Plateau=('Plateau', 'mean'),   # average stiffness across all scan points
        Diagnosis=('Diagnosis', 'first'),   # same label for every row of this patient
    ).reset_index()
    return agg


def get_models():
    """
    Define the 8 classifiers to compare. Each is a scikit-learn pipeline that
    first standardizes the input (zero mean, unit variance) then applies the
    classifier. Standardization is important for distance-based and
    regularization-based models (LR, Ridge, SVM, KNN) which are sensitive to
    the numeric scale of the input.

    Regularization settings (C=0.5, alpha=2.0, max_depth=2):
      These are tighter than sklearn defaults. With only 29 training patients
      per LOOCV fold, overfitting is a real risk. Higher regularization
      forces simpler decision boundaries and reduces the chance the model
      memorizes the training patients instead of learning a generalizable rule.

    class_weight='balanced':
      Even though Controls=15 and SKC=15 are perfectly balanced in this
      dataset, including this flag is good practice for medical classifiers
      to ensure neither class is silently deprioritized if the dataset
      composition ever changes.

    Returns:
        List of (name, pipeline) tuples ready for cross_val_score.
    """
    models = []

    # Logistic Regression: linear decision boundary, probabilistic output.
    # Good baseline for low-dimensional data.
    models.append(('LR',
        make_pipeline(StandardScaler(),
                      LogisticRegression(solver='liblinear', C=0.5,
                                         class_weight='balanced'))))

    # Ridge Classifier: linear like LR but uses a squared penalty instead of
    # L1/L2 log-loss. Often produces comparable results to LR.
    models.append(('Ridge',
        make_pipeline(StandardScaler(),
                      RidgeClassifier(class_weight='balanced', alpha=2.0))))

    # Support Vector Machine with a linear kernel: maximizes the margin
    # between the two classes. Theoretically well-suited for small, clean
    # datasets with one strong feature.
    models.append(('SVM',
        make_pipeline(StandardScaler(),
                      SVC(kernel='linear', class_weight='balanced', C=0.5))))

    # K-Nearest Neighbors: classifies each patient by looking at the 5 most
    # similar training patients (by Mean_Plateau distance). Simple but
    # sensitive to the exact training set composition at small N.
    models.append(('KNN',
        make_pipeline(StandardScaler(),
                      KNeighborsClassifier(n_neighbors=5))))

    # Naive Bayes: assumes each class follows a Gaussian distribution over the
    # feature. Requires no hyperparameter tuning and is a useful sanity check.
    models.append(('NB', GaussianNB()))

    # Decision Tree (CART): finds a single threshold on Mean_Plateau that best
    # splits Controls vs SKC. max_depth=2 prevents it from creating deep trees
    # that memorize training patients (one split per feature at this depth).
    models.append(('CART',
        DecisionTreeClassifier(class_weight='balanced',
                               max_depth=2, min_samples_leaf=3)))

    # Random Forest: trains 50 independent decision trees on random subsets of
    # the training data, then majority-votes their predictions. The max_depth=2
    # constraint keeps each tree simple and prevents overfitting at N=30.
    models.append(('RF',
        RandomForestClassifier(class_weight='balanced', random_state=1,
                               max_depth=2, n_estimators=50)))

    # Extra Trees: similar to Random Forest but uses random split thresholds
    # instead of optimizing them. Often slightly faster and more regularized.
    models.append(('ET',
        ExtraTreesClassifier(class_weight='balanced', random_state=1,
                             max_depth=2, n_estimators=50)))

    return models


def main():

    # ── LOAD DATA ─────────────────────────────────────────────────────────
    # combined_plateau.xlsx is produced by running dataAutomation.py first.
    # It contains one row per spatial scan point across all 30 patients.
    try:
        df = pd.read_excel(INPUT_FILE)
    except FileNotFoundError:
        print(f"Error: Could not find {INPUT_FILE}. Run dataAutomation.py first.")
        return

    # Ensure Patient IDs are treated as strings, not integers, so groupby
    # works correctly even for purely numeric IDs like 20211014.
    df['Patient'] = df['Patient'].astype(str)
    print(f"Loaded {len(df)} spatial points across {df['Patient'].nunique()} patients.")

    # ── HANDLE UNMATCHED PATIENTS ──────────────────────────────────────────
    # dataAutomation.py returns 'Unknown' for any patient file it could not
    # match to the SKC_names.xlsx diagnosis key. Drop those rows so they do
    # not contaminate training with an undefined label.
    n_unknown = (df['Diagnosis'] == 'Unknown').sum()
    if n_unknown:
        print(f"Warning: {n_unknown} rows have Diagnosis='Unknown' and will be dropped.")
        df = df[df['Diagnosis'] != 'Unknown']

    # ── AGGREGATE PER PATIENT ──────────────────────────────────────────────
    # Collapse ~30–44 per-point rows per patient into ONE row: the mean
    # Brillouin plateau shift across that patient's entire scan.
    # This is the key design choice of Algorithm 1: no spatial information,
    # just overall average stiffness per patient.
    patient_table = build_patient_table(df)
    patient_table.to_excel(OUTPUT_FILE, index=False)

    print(f"\nAggregated to {len(patient_table)} patients (1 row each).")
    print(f"Class distribution:\n{patient_table['Diagnosis'].value_counts().to_string()}\n")
    print(f"Saved aggregated table to {OUTPUT_FILE}\n")

    # Print the actual per-patient mean values so you can verify the data
    # looks reasonable before trusting the classifier results.
    print("Mean Plateau per patient:")
    print(patient_table[['Patient', 'Mean_Plateau', 'Diagnosis']].to_string(index=False))
    print()

    # ── RUN CLASSIFIERS WITH LEAVE-ONE-OUT CROSS-VALIDATION ───────────────
    # X: the feature matrix — one row per patient, one column (Mean_Plateau).
    # y: the target labels — 'Controls' or 'SKC' for each patient.
    X = patient_table[FEATURES]
    y = patient_table['Diagnosis']

    models = get_models()

    # LeaveOneOut: at each fold, one patient is held out as the test set and
    # the model trains on the remaining N-1 patients. With 30 patients this
    # means 30 folds, each testing on exactly one patient. The accuracy is the
    # fraction of those 30 individual predictions that are correct.
    loo = LeaveOneOut()

    print(f"Features used: {FEATURES}")
    print("── LEAVE-ONE-PATIENT-OUT CV ──")
    print("   (trains on N-1 patients, tests on 1, repeated for every patient)")

    for name, model in models:
        try:
            # cross_val_score returns a binary array: 1 if correct, 0 if not,
            # one entry per LOOCV fold (i.e., per patient).
            scores = cross_val_score(model, X, y, cv=loo, scoring='accuracy')
            acc = scores.mean()         # fraction of the 30 patients correctly classified
            n   = len(scores)           # always 30 for this dataset

            # 95% confidence interval via normal approximation.
            # At N=30, one wrong prediction swings accuracy by 3.3 points, so
            # CIs are wide (~±10%). Always report the CI floor, not just the
            # headline accuracy, when presenting these results.
            se = np.sqrt(acc * (1 - acc) / n) if 0 < acc < 1 else 0
            print(f"{name:5s}: {acc*100:5.1f}%  ({int(round(acc*n))}/{n} correct)  "
                  f"approx 95% CI {max(0,(acc-1.96*se))*100:4.1f}%"
                  f"-{min(1,(acc+1.96*se))*100:4.1f}%")
        except Exception as e:
            print(f"{name}: ERROR - {e}")


if __name__ == '__main__':
    main()