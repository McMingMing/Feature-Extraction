"""
Medical Diagnosis Classifier - Algorithm 2
===========================================================
This script trains and evaluates multiple classification models to predict patient diagnoses
based on THREE features
  1. Plateau
  2. X position (mm)
  3. Y position (mm)

Key approaches:
1. Multi-feature input: Uses Plateau, X, and Y
2. Group-based splitting: Prevents patient data leakage using GroupShuffleSplit
3. Multiple model comparison: Tests 11 algorithms to find best performers
4. Dual evaluation:
   - Single Split (80/20): Quick diagnostic for overfitting
   - Leave-One-Patient-Out (LOGO): Primary validation for real generalization
"""

import pandas as pd
from sklearn.ensemble import BaggingClassifier, ExtraTreesClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import GroupShuffleSplit, LeaveOneGroupOut, cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# ==================== DATA LOADING & PREPROCESSING ====================
# WHY dropna: Some measurements may be missing or invalid.
# We remove incomplete rows to ensure clean training data.
# With more features, we're more likely to have some missing values.
COMBINED_FILE = '/Users/minhnguyen/try-scipy/combined_data.xlsx'
dataset = pd.read_excel(COMBINED_FILE)
dataset = dataset.dropna(subset=['Plateau', 'X (mm)', 'Y (mm)'])
print(f"Total measurements after removing missing values: {len(dataset)}")

# Prepare features and target
# X: three features (Plateau + spatial coordinates X and Y)
# y: diagnosis label
# groups: patient IDs (critical for preventing patient-level data leakage)
X = dataset[['Plateau', 'X (mm)', 'Y (mm)']]
y = dataset['Diagnosis']
groups = dataset['Patient']

# ==================== TRAIN/TEST SPLIT ====================
# WHY GroupShuffleSplit with groups parameter:
# Each patient typically has multiple measurements at different X and Y locations.
# If we randomly split measurements, the SAME PATIENT could end up in both
# training and validation → DATA LEAKAGE → artificially inflated accuracy.
#
# Solution: Use GroupShuffleSplit to split by PATIENT GROUP, not by measurements.
# This ensures every patient is either 100% in training OR 100% in validation.
#
# Note: random_state=3 (different seed from algorithm1 for variety)
gss = GroupShuffleSplit(n_splits=1, train_size=0.8, random_state=3)
train_idx, test_idx = next(gss.split(X, y, groups))
X_train, X_validation = X.iloc[train_idx], X.iloc[test_idx]
Y_train, Y_validation = y.iloc[train_idx], y.iloc[test_idx]

# Display class distribution
print("── TRAINING DIAGNOSIS BREAKDOWN ──")
print(Y_train.value_counts())
print("\n── VALIDATION DIAGNOSIS BREAKDOWN ──")
print(Y_validation.value_counts())
print("───────────────────────────────────\n")

# ==================== MODEL SELECTION ====================
# We test 11 candidate models because:
# 1. Different algorithms have different strengths for multi-feature problems
# 2. With more features (3 instead of 1), ensemble models may perform better
# 3. Regularization and hyperparameters are tuned to reduce overfitting risk
#
# Changes from algorithm1:
# - Added StandardScaler pipeline for LR and SVM (important with multiple features)
# - Added tree depth limits to prevent overfitting
# - Added min_samples_leaf to ensure splits are meaningful
# - Tuned ensemble sizes and sampling rates
models = []
models.append(('LR', make_pipeline(
    StandardScaler(),
    LogisticRegression(solver='liblinear', C=1.0, class_weight='balanced')
)))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier(n_neighbors=3)))
models.append(('CART', DecisionTreeClassifier(class_weight='balanced', max_depth=3, min_samples_leaf=2)))
models.append(('NB', GaussianNB()))
models.append(('SVM', make_pipeline(
    StandardScaler(),
    SVC(gamma='scale', class_weight='balanced', C=1.0)
)))
models.append(('RF', RandomForestClassifier(class_weight='balanced', random_state=1, max_depth=4, min_samples_leaf=2, n_estimators=100)))
models.append(('GB', GradientBoostingClassifier(random_state=1, max_depth=3)))
models.append(('ET', ExtraTreesClassifier(class_weight='balanced', random_state=1, max_depth=4, min_samples_leaf=2, n_estimators=100)))
models.append(('Ridge', RidgeClassifier(class_weight='balanced', alpha=1.0)))
models.append(('Bag', BaggingClassifier(random_state=1, n_estimators=25, max_samples=0.8)))

# ==================== SINGLE SPLIT EVALUATION ====================
# The 80/20 split is a quick diagnostic, not the primary validation metric.
#
# Train vs Validation Gap interpretation:
#   - Small gap (~< 10%): usually indicates good generalization on larger datasets
#   - Moderate gap (10-20%): caution needed, especially with small datasets
#   - Large gap (> 20%): suggests overfitting or a difficult validation split
#   - Negative gap: rare, may indicate validation set is easier than training
#
# IMPORTANT: Leave-One-Patient-Out is the primary generalization test for this dataset.
#           Single split results are informative but secondary.
print("── SINGLE SPLIT RESULTS (Train vs Validation) ──")
for name, model in models:
    try:
        model.fit(X_train, Y_train)
        train_score = accuracy_score(Y_train, model.predict(X_train))
        val_score = accuracy_score(Y_validation, model.predict(X_validation))
        gap = train_score - val_score
        print(f"{name}:  Train: {train_score*100:.1f}%  |  Validation: {val_score*100:.1f}%  |  Gap: {gap*100:.1f}%")
    except Exception as e:
        print(f"{name}: ERROR — {e}")

# ==================== LEAVE-ONE-PATIENT-OUT CROSS-VALIDATION ====================
# Leave-One-Patient-Out (LOGO) is the gold standard for small medical datasets.
#
# How it works:
#   - On each iteration, ONE COMPLETE PATIENT is held out as test
#   - All OTHER patients (with all their measurements) are used for training
#   - Repeat until every patient has been a test case exactly once
#   - Report average accuracy across all iterations
#
# Why LOGO is best for this problem:
#   - Uses MAXIMUM training data per fold (~90% of patients)
#   - Completely eliminates within-patient data leakage
#   - Gives the most realistic estimate of how the model will perform on
#     brand-new unseen patients
#   - Essential for medical applications where generalization is critical
#
# Interpretation:
#   - LOGO accuracy is the best estimate of real-world performance
#   - Usually lower than single-split validation (smaller training sets per fold)
#   - But more honest and less biased
logo = LeaveOneGroupOut()

print("\n── LEAVE-ONE-PATIENT-OUT RESULTS ──")
for name, model in models:
    try:
        # cross_val_score handles all train/test splitting automatically
        # Returns an array of accuracy scores, one per patient left out
        scores = cross_val_score(model, X, y, cv=logo, groups=groups, scoring='accuracy')
        print(f"{name}: {scores.mean():.4f} ({scores.mean()*100:.1f}% avg accuracy)")
    except Exception as e:
        print(f"{name}: ERROR — {e}")