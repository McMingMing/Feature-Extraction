"""
Brillouin Cornea DL - Data Loader (central 2mm spatial filter)
==============================================================
Builds the deep-learning dataset from the per-patient .mat files,
restricting to spatial points within 2mm of the cornea center.

WHY SPATIAL FILTERING
---------------------
The PI's published paper establishes the central ~2mm region as the
clinically relevant zone for keratoconus diagnosis. While peripheral
points also carry statistical signal (Cohen's d ~1.78 beyond 3mm),
the scientific argument for the diagnosis is grounded in the center.
We train on the center-restricted dataset first, then on all points,
to make that comparison argument directly.

HOW THE FILTER WORKS
--------------------
combined_data.xlsx maps each spatial point to its X/Y coordinates in mm.
The column index in shifts_before.mat corresponds 1:1 to the row index
in combined_data.xlsx for that patient. We compute the radial distance
sqrt(X^2 + Y^2) for each point and keep only those within RADIUS_MM.

ONE TRAINING SAMPLE = one spatial point's depth profile (within 2mm)
= one column of the .mat array = 100 Brillouin-shift values down Z-axis.

INPUT  : DATA_DIR/<patient_id>/shifts_before.mat
         COORD_FILE (combined_data.xlsx) for X/Y coordinate lookup
OUTPUT : dl_dataset_2mm.npz  (n_sequences, 100, 1)
"""

import os
import glob
import numpy as np
import pandas as pd
import scipy.io

# ── CONFIGURATION ─────────────────────────────────────────────────────────
DATA_DIR    = '/Users/minhnguyen/deep-learning/All Brillouin Point Data'
COORD_FILE  = '/Users/minhnguyen/deep-learning/combined_data.xlsx'
MAT_FILENAME = 'shifts_before.mat'
MAT_KEY      = 'shifts_before'
OUTPUT_NPZ   = '/Users/minhnguyen/deep-learning/dl_dataset_2mm.npz'

RADIUS_MM  = 2.0    # only keep points within this radius of cornea center
LABEL_MAP  = {'Controls': 0, 'SKC': 1}
# ──────────────────────────────────────────────────────────────────────────


def load_coord_map(path):
    """
    Returns a dict: {patient_id: DataFrame with columns [radius, Diagnosis]}
    indexed by the point's position within that patient (0-based), so we can
    look up whether column i of the .mat file is within 2mm.
    """
    df = pd.read_excel(path)
    df['Patient'] = df['Patient'].astype(str)
    df['radius'] = np.sqrt(df['X (mm)']**2 + df['Y (mm)']**2)

    coord_map = {}
    for pid, grp in df.groupby('Patient'):
        # Reset index so position 0,1,2... matches .mat column 0,1,2...
        coord_map[pid] = grp[['radius', 'Diagnosis']].reset_index(drop=True)
    return coord_map


def build_dataset():
    coord_map = load_coord_map(COORD_FILE)
    print(f"Loaded coordinates for {len(coord_map)} patients.")
    print(f"Spatial filter: keeping points within {RADIUS_MM}mm of center.\n")

    sequences, labels, groups = [], [], []
    skipped = []

    patient_dirs = sorted(glob.glob(os.path.join(DATA_DIR, '*')))
    patient_dirs = [p for p in patient_dirs if os.path.isdir(p)]

    for pdir in patient_dirs:
        patient_id = os.path.basename(pdir)
        mat_path = os.path.join(pdir, MAT_FILENAME)

        if not os.path.exists(mat_path):
            skipped.append((patient_id, 'no .mat file'))
            continue

        # Get coordinate info for this patient
        coords = coord_map.get(patient_id)
        if coords is None:
            # Try date prefix only (handles "20220628 Left" vs "20220628")
            base = patient_id.split()[0]
            coords = coord_map.get(base)
        if coords is None:
            skipped.append((patient_id, 'no coordinate entry'))
            continue

        # Get diagnosis from the coordinate map
        diagnosis = coords['Diagnosis'].iloc[0]
        if diagnosis not in LABEL_MAP:
            skipped.append((patient_id, f'unknown diagnosis: {diagnosis}'))
            continue
        label = LABEL_MAP[diagnosis]

        # Load .mat array (100 x N)
        mat = scipy.io.loadmat(mat_path)
        if MAT_KEY not in mat:
            skipped.append((patient_id, f'key {MAT_KEY} missing'))
            continue
        arr = mat[MAT_KEY]

        if arr.shape[0] != 100:
            print(f"  Note: {patient_id} has {arr.shape[0]} depth rows, expected 100.")

        n_points = arr.shape[1]

        # Verify column count matches coordinate rows
        if n_points != len(coords):
            print(f"  Warning: {patient_id} has {n_points} .mat columns but "
                  f"{len(coords)} coordinate rows. Using min({n_points},{len(coords)}).")
            n_points = min(n_points, len(coords))

        n_kept = 0
        for col in range(n_points):
            # Apply spatial filter using the coordinate map
            r = coords['radius'].iloc[col]
            if r > RADIUS_MM:
                continue

            seq = arr[:, col].astype(np.float32)
            if np.isnan(seq).any():
                continue

            sequences.append(seq)
            labels.append(label)
            groups.append(patient_id)
            n_kept += 1

        print(f"  {patient_id:15s} {diagnosis:9s}  "
              f"{n_kept}/{n_points} points within {RADIUS_MM}mm kept")

    if not sequences:
        print("\nNo sequences loaded. Check DATA_DIR and COORD_FILE paths.")
        return

    X = np.stack(sequences)[..., np.newaxis]  # (n, 100, 1)
    y = np.array(labels, dtype=np.int64)
    groups = np.array(groups)

    np.savez(OUTPUT_NPZ, X=X, y=y, groups=groups)

    print("\n" + "=" * 60)
    print(f"Saved: {OUTPUT_NPZ}")
    print(f"X shape: {X.shape}   y shape: {y.shape}")
    print(f"Patients: {len(np.unique(groups))}")
    print(f"Sequences per class:  "
          f"Controls={int((y==0).sum())}  SKC={int((y==1).sum())}")
    if skipped:
        print(f"\nSkipped {len(skipped)} patient folder(s):")
        for pid, reason in skipped:
            print(f"  {pid}: {reason}")
    print("=" * 60)


if __name__ == '__main__':
    build_dataset()