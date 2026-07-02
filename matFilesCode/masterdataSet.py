"""
Brillouin Cornea ML - Feature Builder (.mat-based pipeline)
===========================================================
This script reads the per-point plateau data in combined_data.xlsx (one row per
spatial point, with X/Y coordinates and the Lorentzian-fitted plateau value) and
aggregates each patient down to a single row of ML features.

WHY THIS DESIGN
---------------
The discriminating signal between Subclinical Keratoconus (SKC) and Controls is
NOT just a lower mean stiffness. As the Brillouin maps in the reference paper show,
a healthy cornea is spatially UNIFORM, while an SKC cornea is softer in the center
and stiffer at the periphery. So the features below capture both the magnitude of
stiffness AND its spatial distribution.

INPUT  : combined_data.xlsx   (columns: Patient, Plateau, X (mm), Y (mm), Diagnosis)
OUTPUT : ML_Master_Dataset.xlsx  (one row per patient, engineered features + Diagnosis)

Coordinate convention: radius = sqrt(X^2 + Y^2), measured in mm from cornea center.
  - inner ring  : radius <  1 mm   (very central)
  - middle ring : 1 <= radius <= 2 mm
  - "within 2mm": radius <= 2 mm   (inner + middle, the mentor's region of interest)
"""

import pandas as pd
import numpy as np

# ── CONFIGURATION ─────────────────────────────────────────────────────────
INPUT_FILE  = '/Users/minhnguyen/try-scipy/combined_data.xlsx'
OUTPUT_FILE = '/Users/minhnguyen/try-scipy/ML_Master_Dataset.xlsx'

INNER_RADIUS  = 1.0   # mm - boundary of the central ring
REGION_RADIUS = 2.0   # mm - boundary of the mentor's "region of interest"
# ──────────────────────────────────────────────────────────────────────────


def build_features_for_patient(group):
    """Aggregate one patient's spatial points into a single feature row.

    `group` is all rows in combined_data.xlsx belonging to one Patient ID.
    Each row is one spatial measurement point on that patient's cornea.
    """
    plateau = group['Plateau'].values
    radius  = group['radius'].values

    # --- Masks for the spatial rings ---
    within_2mm = radius <= REGION_RADIUS
    inner_ring = radius < INNER_RADIUS
    mid_ring   = (radius >= INNER_RADIUS) & (radius <= REGION_RADIUS)

    p_within = plateau[within_2mm]
    p_inner  = plateau[inner_ring]
    p_mid    = plateau[mid_ring]

    features = {}

    # FEATURE 1: Mean plateau within 2mm (mentor's primary suggestion).
    #   The average corneal stiffness in the clinically relevant central region.
    features['Mean_Plateau_2mm'] = np.mean(p_within) if len(p_within) else np.nan

    # FEATURE 2: Mean plateau of ALL points (mentor: "try with and without" the
    #   2mm filter). Lets us compare whether restricting to the center helps.
    features['Mean_Plateau_All'] = np.mean(plateau)

    # FEATURE 3: Std of plateau within 2mm (UNIFORMITY).
    #   Low std = uniform stiffness = healthy. High std = irregular = possible SKC.
    features['Std_Plateau_2mm'] = np.std(p_within) if len(p_within) else np.nan

    # FEATURE 4: Center-to-periphery gradient.
    #   = mean(inner ring) - mean(middle ring).
    #   This is the spatial signature the Brillouin maps show: in SKC the center
    #   is softer than the surround, so this value tends to go NEGATIVE.
    #   In a uniform healthy cornea it sits near zero.
    if len(p_inner) and len(p_mid):
        features['Center_Periphery_Gradient'] = np.mean(p_inner) - np.mean(p_mid)
    else:
        features['Center_Periphery_Gradient'] = np.nan

    # FEATURE 5: Min plateau within 2mm.
    #   Captures a localized soft spot, which is what a keratoconus cone produces.
    features['Min_Plateau_2mm'] = np.min(p_within) if len(p_within) else np.nan

    # --- Bookkeeping (not features, but useful for sanity-checking) ---
    features['N_Points_Total'] = len(plateau)
    features['N_Points_2mm']   = int(within_2mm.sum())
    features['Diagnosis']      = group['Diagnosis'].iloc[0]

    return pd.Series(features)


def main():
    # 1. Load the per-point data
    try:
        df = pd.read_excel(INPUT_FILE)
    except FileNotFoundError:
        print(f"Error: Could not find {INPUT_FILE}. Check the path.")
        return

    print(f"Loaded {len(df)} spatial points across {df['Patient'].nunique()} patients.\n")

    # 2. Compute the radial distance for every point
    df['radius'] = np.sqrt(df['X (mm)']**2 + df['Y (mm)']**2)

    # 3. Aggregate each patient into one feature row
    master = df.groupby('Patient').apply(build_features_for_patient).reset_index()

    # 4. Report any patients with missing features (e.g. empty ring)
    feature_cols = ['Mean_Plateau_2mm', 'Mean_Plateau_All', 'Std_Plateau_2mm',
                    'Center_Periphery_Gradient', 'Min_Plateau_2mm']
    n_missing = master[feature_cols].isna().any(axis=1).sum()
    if n_missing:
        print(f"Warning: {n_missing} patient(s) have a missing feature (empty ring).")
        print(master[master[feature_cols].isna().any(axis=1)][['Patient'] + feature_cols])
        print()

    # 5. Save
    master.to_excel(OUTPUT_FILE, index=False)

    print("═" * 60)
    print(f"Master dataset built: {OUTPUT_FILE}")
    print(f"Patients: {len(master)}")
    print(f"Class distribution:\n{master['Diagnosis'].value_counts().to_string()}")
    print("═" * 60)
    print("\nFeature preview:")
    print(master[['Patient'] + feature_cols + ['Diagnosis']].to_string(index=False))


if __name__ == '__main__':
    main()