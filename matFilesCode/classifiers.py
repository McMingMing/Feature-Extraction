"""
Brillouin Cornea ML - Classifier Comparison (.mat-based pipeline)
================================================================
Reads the patient-level feature sheet built by buildMasterDataset.py and
compares classifiers for separating Subclinical Keratoconus (SKC) from Controls.

KEY METHODOLOGY NOTES FOR N=30
------------------------------
- Each patient is now ONE row (not many points), so there is no patient-level
  leakage to worry about: a standard split already keeps a patient whole.
  Leave-One-Out CV here means leave-one-PATIENT-out, the honest estimate.
- With only 30 patients, every accuracy number has a wide confidence interval.
  A single misclassification swings leave-one-out accuracy by 3.3 percentage
  points. Treat differences of a few percent between models as noise.
- We deliberately keep feature count low and regularization high to avoid
  memorizing 30 patients.

FEATURE_SETS lets you swap which features go into the model, so you can test
the mentor's "with and without the 2mm average" question directly.

OUTPUT EXPLAINED
----------------
5-Fold Stratified CV:
  Splits 30 patients into 5 groups of 6. Trains on 24, tests on 6, repeats
  5 times so every patient is tested once. The percentage is the average
  accuracy across all 5 rounds. The std shows how much accuracy varied
  between rounds. A high std (e.g. 13.3%) just means one bad fold of 6
  patients can swing results significantly - this is expected at N=30.

Leave-One-Patient-Out CV:
  Trains on 29 patients, tests on 1, repeats 30 times. Every patient gets
  to be the sole test case exactly once. The percentage = fraction of those
  30 individual predictions that were correct. This is the most honest
  accuracy estimate for small medical datasets and the number to report.
  Example: 93.3% = 28 out of 30 patients correctly classified.

95% Confidence Interval:
  Because N=30 is small, the true accuracy on new unseen patients could
  differ from what we measure here. The CI gives the plausible range.
  Example: NB at 96.7% (CI 90.2%-100%) means the real-world accuracy
  is probably somewhere between 90% and 100%, not necessarily 96.7%.
  Always report the CI floor, not just the point estimate.
"""

import pandas as pd
import numpy as np
import warnings
from sklearn.model_selection import LeaveOneOut, StratifiedKFold, cross_val_score
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
FILE_PATH = 'ML_Master_Dataset.xlsx'

# Each entry is a named feature set you can test independently.
# Switch ACTIVE_SET to compare different hypotheses:
#   'mentor_2mm_only'  - just the central mean stiffness (mentor's primary suggestion)
#   'mentor_with_avg'  - adds the all-points average (mentor: "try with and without")
#   'spatial'          - mean + uniformity + center-to-periphery gradient
#   'all_features'     - everything, lets the model sort out what matters
FEATURE_SETS = {
    'mentor_2mm_only':   ['Mean_Plateau_2mm'],
    'mentor_with_avg':   ['Mean_Plateau_2mm', 'Mean_Plateau_All'],
    'spatial':           ['Mean_Plateau_2mm', 'Std_Plateau_2mm', 'Center_Periphery_Gradient'],
    'all_features':      ['Mean_Plateau_2mm', 'Mean_Plateau_All', 'Std_Plateau_2mm',
                          'Center_Periphery_Gradient', 'Min_Plateau_2mm'],
}
ACTIVE_SET = 'all_features'   # <-- change this to test different feature combinations
# ──────────────────────────────────────────────────────────────────────────


def get_models():
    """
    Returns all classifiers to compare.

    Why these settings:
    - C=0.5 on LR/SVM: higher regularization forces simpler decision boundaries,
      reducing overfitting on only 30 patients.
    - alpha=2.0 on Ridge: same idea, penalizes large coefficients.
    - max_depth=2 on tree models: prevents trees from memorizing individual patients
      by limiting how many splits they can make.
    - class_weight='balanced': since SKC=15 and Controls=15 are already balanced
      this doesn't change much, but it's good practice for medical classifiers
      so neither class is silently deprioritized.
    """
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
        dataset = pd.read_excel(FILE_PATH)
    except FileNotFoundError:
        print(f"Error: Could not find {FILE_PATH}. Run buildMasterDataset.py first.")
        return

    features = FEATURE_SETS[ACTIVE_SET]
    dataset = dataset.dropna(subset=features + ['Diagnosis'])

    print(f"Active feature set: '{ACTIVE_SET}'  ->  {features}")
    print(f"Patients: {len(dataset)}")
    print(f"Class distribution:\n{dataset['Diagnosis'].value_counts().to_string()}\n")

    X = dataset[features]
    y = dataset['Diagnosis']

    models = get_models()

    # ── 5-FOLD STRATIFIED CV ──────────────────────────────────────────────
    # Splits 30 patients into 5 groups of 6. Trains on 24, tests on 6,
    # repeats 5 times. "Stratified" ensures each fold keeps the 50/50 class
    # ratio so no fold accidentally gets all SKC or all Controls.
    # The std tells you how much accuracy varied between the 5 folds.
    # High std is expected here because 1 wrong prediction per fold = 16.7% swing.
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=3)
    print("── 5-FOLD STRATIFIED CV ──")
    print("   (avg accuracy across 5 rounds, each testing 6 patients)")
    for name, model in models:
        try:
            scores = cross_val_score(model, X, y, cv=skf, scoring='accuracy')
            print(f"{name:5s}: {scores.mean()*100:5.1f}%  (std {scores.std()*100:4.1f}%)")
        except Exception as e:
            print(f"{name}: ERROR - {e}")

    # ── LEAVE-ONE-PATIENT-OUT CV ──────────────────────────────────────────
    # The most honest estimate for small medical datasets.
    # Trains on 29 patients, tests on 1, repeats 30 times.
    # Accuracy = fraction of the 30 individual predictions that were correct.
    # Example: 93.3% means 28 out of 30 patients were correctly classified.
    # The 95% CI shows the plausible range of true accuracy on new patients.
    # Report the CI floor alongside the point estimate, not just the headline number.
    loo = LeaveOneOut()
    print("\n── LEAVE-ONE-PATIENT-OUT CV ──")
    print("   (trains on 29 patients, tests on 1, repeated 30 times)")
    print("   (accuracy = fraction of 30 individual patients correctly classified)")
    for name, model in models:
        try:
            scores = cross_val_score(model, X, y, cv=loo, scoring='accuracy')
            acc = scores.mean()
            # 95% CI via normal approximation: gives a plausible range for the
            # true accuracy if you applied this model to new unseen patients.
            # The floor of the CI is the conservative number to report.
            se  = np.sqrt(acc * (1 - acc) / len(scores))
            print(f"{name:5s}: {acc*100:5.1f}%  "
                  f"({int(acc*30)}/30 correct)  "
                  f"approx 95% CI {max(0,(acc-1.96*se))*100:4.1f}%-{min(1,(acc+1.96*se))*100:4.1f}%")
        except Exception as e:
            print(f"{name}: ERROR - {e}")


if __name__ == '__main__':
    main()