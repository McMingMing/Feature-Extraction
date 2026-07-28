"""
Algorithm 1 - ML Training
======================================

WHAT THIS SCRIPT DOES
----------------------
Takes the output of dataAutomation.py (combined_plateau.xlsx - one row per
spatial point, columns: Patient, Plateau, Diagnosis) and:
  1. Averages the Plateau values per patient (collapsing ~30+ points/patient
     down to one row per patient, since combined_plateau.xlsx has no X/Y
     coordinates and so can't support the spatial 2mm/gradient features from
     the coordinate-based pipeline).
  2. Runs the same 8-classifier comparison used elsewhere in this project,
     validated with Leave-One-Patient-Out CV (the honest estimate at N=30).

INPUT  : combined_plateau.xlsx  (from dataAutomation.py)
OUTPUT : prints classifier comparison to terminal; also saves the per-patient
         aggregated table to Algorithm1_Master_Dataset.xlsx for inspection.
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

warnings.filterwarnings("ignore")

# ── CONFIGURATION ─────────────────────────────────────────────────────────
INPUT_FILE  = 'combined_plateau.xlsx'          # output of dataAutomation.py
OUTPUT_FILE = 'Algorithm1_Master_Dataset.xlsx'  # per-patient aggregated table

# Which aggregated features to feed the classifiers.
# Mentor's original ask: "take the average of the plateau data and run the
# ML code" — so this is just the per-patient mean, nothing else.
FEATURES = ['Mean_Plateau']
# ──────────────────────────────────────────────────────────────────────────


def build_patient_table(df):
    """Collapse the per-point rows down to one row per patient: just the
    mean Plateau value and the Diagnosis label."""
    agg = df.groupby('Patient').agg(
        Mean_Plateau=('Plateau', 'mean'),
        Diagnosis=('Diagnosis', 'first'),
    ).reset_index()
    return agg


def get_models():
    models = []
    models.append(('LR',    make_pipeline(StandardScaler(), LogisticRegression(solver='liblinear', C=0.5, class_weight='balanced'))))
    models.append(('Ridge', make_pipeline(StandardScaler(), RidgeClassifier(class_weight='balanced', alpha=2.0))))
    models.append(('SVM',   make_pipeline(StandardScaler(), SVC(kernel='linear', class_weight='balanced', C=0.5))))
    models.append(('KNN',   make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=5))))
    models.append(('NB',    GaussianNB()))
    models.append(('CART',  DecisionTreeClassifier(class_weight='balanced', max_depth=2, min_samples_leaf=3)))
    models.append(('RF',    RandomForestClassifier(class_weight='balanced', random_state=1, max_depth=2, n_estimators=50)))
    models.append(('ET',    ExtraTreesClassifier(class_weight='balanced', random_state=1, max_depth=2, n_estimators=50)))
    return models


def main():
    try:
        df = pd.read_excel(INPUT_FILE)
    except FileNotFoundError:
        print(f"Error: Could not find {INPUT_FILE}. Run dataAutomation.py first.")
        return

    df['Patient'] = df['Patient'].astype(str)
    print(f"Loaded {len(df)} spatial points across {df['Patient'].nunique()} patients.")

    # Filter out any rows where diagnosis lookup failed (dataAutomation.py
    # returns 'Unknown' when a filename couldn't be matched to the SKC list).
    n_unknown = (df['Diagnosis'] == 'Unknown').sum()
    if n_unknown:
        print(f"Warning: {n_unknown} rows have Diagnosis='Unknown' and will be dropped.")
        df = df[df['Diagnosis'] != 'Unknown']

    # ── AGGREGATE: average (and std) plateau per patient ───────────────────
    patient_table = build_patient_table(df)
    patient_table.to_excel(OUTPUT_FILE, index=False)

    print(f"\nAggregated to {len(patient_table)} patients (1 row each).")
    print(f"Class distribution:\n{patient_table['Diagnosis'].value_counts().to_string()}\n")
    print(f"Saved aggregated table to {OUTPUT_FILE}\n")

    # ── RUN CLASSIFIERS ──────────────────────────────────────────────────
    X = patient_table[FEATURES]
    y = patient_table['Diagnosis']

    models = get_models()
    loo = LeaveOneOut()
    
    print(f"\nAggregated to {len(patient_table)} patients (1 row each).")
    print(f"Class distribution:\n{patient_table['Diagnosis'].value_counts().to_string()}\n")
    print(f"Saved aggregated table to {OUTPUT_FILE}\n")

    print("Mean Plateau per patient:")
    print(patient_table[['Patient', 'Mean_Plateau', 'Diagnosis']].to_string(index=False))
    print()

    # ── RUN CLASSIFIERS ──────────────────────────────────────────────────
    print(f"Features used: {FEATURES}")
    print("── LEAVE-ONE-PATIENT-OUT CV ──")
    print("   (trains on N-1 patients, tests on 1, repeated for every patient)")
    for name, model in models:
        try:
            scores = cross_val_score(model, X, y, cv=loo, scoring='accuracy')
            acc = scores.mean()
            n = len(scores)
            se = np.sqrt(acc * (1 - acc) / n) if 0 < acc < 1 else 0
            print(f"{name:5s}: {acc*100:5.1f}%  ({int(round(acc*n))}/{n} correct)  "
                  f"approx 95% CI {max(0,(acc-1.96*se))*100:4.1f}%-{min(1,(acc+1.96*se))*100:4.1f}%")
        except Exception as e:
            print(f"{name}: ERROR - {e}")


if __name__ == '__main__':
    main()