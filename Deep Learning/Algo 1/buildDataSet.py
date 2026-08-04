"""
Brillouin Cornea DL - Data Loader (.mat depth-profile sequences)
================================================================
Builds the deep-learning dataset from the per-patient .mat files. Unlike the ML
pipeline (which aggregated each patient to ONE row), the DL pipeline keeps every
spatial point as its own training sequence.

ONE TRAINING SAMPLE = one spatial point's depth profile = one column of the
.mat array = a sequence of 100 Brillouin-shift values down the Z (depth) axis.
This matches Ron's LSTM template, where time_step=100 and input_dim=1.

CRITICAL DESIGN POINT - PATIENT-LEVEL GROUPING
----------------------------------------------
We have ~1081 sequences but only 30 patients. Sequences from the same patient
are highly correlated. If train/test splitting mixes a patient's sequences across
both sides, the model memorizes that patient and reports a fake-high accuracy.
So this loader returns a `groups` array (the patient ID for every sequence). The
training script MUST split on those groups, never on raw sequence indices.

INPUT  : - DATA_DIR/<patient_id>/shifts_before.mat   (100 x N array per patient)
         - combined_data.xlsx                        (patient -> diagnosis map)
OUTPUT : X (n_sequences, 100, 1), y (n_sequences,), groups (n_sequences,)
         saved to dl_dataset.npz for the training script to load.
"""

import os
import glob
import numpy as np
import pandas as pd
import scipy.io

# ── CONFIGURATION ─────────────────────────────────────────────────────────
DATA_DIR       = '/Users/minhnguyen/deep-learning/All Brillouin Point Data'
DIAGNOSIS_FILE = '/Users/minhnguyen/deep-learning/combined_data.xlsx'
MAT_FILENAME   = 'shifts_before.mat'
MAT_KEY        = 'shifts_before'
OUTPUT_NPZ = '/Users/minhnguyen/deep-learning/dl_dataset.npz'
# Map the two diagnosis strings to integer labels.
LABEL_MAP = {'Controls': 0, 'SKC': 1}
# ──────────────────────────────────────────────────────────────────────────


def load_diagnosis_map(path):
    """Return {patient_id_string: diagnosis_string} from the combined sheet."""
    df = pd.read_excel(path)
    # One diagnosis per patient; first() collapses the repeated rows.
    diag = df.groupby(df['Patient'].astype(str))['Diagnosis'].first()
    return diag.to_dict()


def build_dataset():
    diagnosis_map = load_diagnosis_map(DIAGNOSIS_FILE)
    print(f"Loaded diagnoses for {len(diagnosis_map)} patients.\n")

    sequences = []   # each: (100,) depth profile
    labels    = []   # 0 or 1
    groups    = []   # patient id string

    # Each patient is a subfolder of DATA_DIR.
    patient_dirs = sorted(glob.glob(os.path.join(DATA_DIR, '*')))
    patient_dirs = [p for p in patient_dirs if os.path.isdir(p)]

    skipped = []
    for pdir in patient_dirs:
        patient_id = os.path.basename(pdir)
        mat_path = os.path.join(pdir, MAT_FILENAME)

        if not os.path.exists(mat_path):
            skipped.append((patient_id, 'no .mat file'))
            continue

        # Resolve diagnosis. Try the full folder name, then the date prefix
        # (handles names like "20220628 Left" vs "20220628").
        diagnosis = diagnosis_map.get(patient_id)
        if diagnosis is None:
            diagnosis = diagnosis_map.get(patient_id.split()[0])
        if diagnosis is None or diagnosis not in LABEL_MAP:
            skipped.append((patient_id, f'no usable diagnosis ({diagnosis})'))
            continue

        label = LABEL_MAP[diagnosis]

        # Load the 100 x N array. Columns are spatial points, rows are depth.
        mat = scipy.io.loadmat(mat_path)
        if MAT_KEY not in mat:
            skipped.append((patient_id, f'key {MAT_KEY} missing'))
            continue
        arr = mat[MAT_KEY]  # shape (100, N)

        if arr.shape[0] != 100:
            # Don't silently reshape; report it so you can investigate.
            print(f"  Note: {patient_id} has {arr.shape[0]} depth rows, expected 100.")

        n_points = arr.shape[1]
        for col in range(n_points):
            seq = arr[:, col].astype(np.float32)
            # Drop any sequence with NaNs rather than feeding garbage to the net.
            if np.isnan(seq).any():
                continue
            sequences.append(seq)
            labels.append(label)
            groups.append(patient_id)

        print(f"  {patient_id:15s} {diagnosis:9s}  +{n_points} sequences")

    if not sequences:
        print("\nNo sequences loaded. Check DATA_DIR and file names.")
        return

    X = np.stack(sequences)               # (n, 100)
    X = X[..., np.newaxis]                 # (n, 100, 1) for the LSTM
    y = np.array(labels, dtype=np.int64)   # (n,)
    groups = np.array(groups)              # (n,)

    np.savez(OUTPUT_NPZ, X=X, y=y, groups=groups)

    print("\n" + "=" * 60)
    print(f"Saved {OUTPUT_NPZ}")
    print(f"X shape: {X.shape}   y shape: {y.shape}")
    print(f"Patients: {len(np.unique(groups))}")
    print(f"Sequences per class: Controls={int((y==0).sum())}  SKC={int((y==1).sum())}")
    if skipped:
        print(f"\nSkipped {len(skipped)} patient folder(s):")
        for pid, reason in skipped:
            print(f"  {pid}: {reason}")
    print("=" * 60)


if __name__ == '__main__':
    build_dataset()