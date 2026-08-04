"""
Brillouin Cornea DL - LSTM Classifier (adapted from Ron's template)
==================================================================
Trains an LSTM to classify a single spatial point's depth profile as
Controls vs SKC, then aggregates point-level predictions up to a
patient-level diagnosis for an apples-to-apples comparison with the ML pipeline.

ARCHITECTURE: follows the structure of Ron's template (stacked LSTM layers ->
dense output) but adapted in three ways for THIS problem:
  1. Binary task: final layer is 1 unit + sigmoid (not 5-way softmax).
  2. Right-sized: Ron's template used units = 12*batch_size = 192 units per
     layer. Three 192-unit LSTM layers on ~1000 short sequences will memorize
     the training set. We shrink the units substantially. You can dial this
     back up with UNITS if you want to reproduce his exact width, but watch the
     train-vs-validation gap when you do.
  3. Normalized input: Brillouin shifts are ~4e9 Hz. LSTMs train terribly on
     raw values that large, so we standardize. The scaler is fit on TRAIN ONLY.

CRITICAL: PATIENT-LEVEL SPLIT
-----------------------------
We split by patient, not by sequence. A patient's ~36 sequences are correlated;
letting some land in train and some in test would leak identity and inflate
accuracy. GroupShuffleSplit on the `groups` array guarantees every patient is
wholly in train or wholly in test.
"""

import numpy as np
import tensorflow as tf
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix

# ── CONFIGURATION ─────────────────────────────────────────────────────────
DATASET_NPZ = 'dl_dataset.npz'
UNITS       = 64      # per-LSTM-layer width. Ron's template implied 192; bumped from 32.
EPOCHS      = 50
BATCH_SIZE  = 16
TEST_FRAC   = 0.2     # fraction of PATIENTS held out for testing
SEED        = 1
# ──────────────────────────────────────────────────────────────────────────

tf.random.set_seed(SEED)
np.random.seed(SEED)


def build_model(time_step, units):
    """Stacked LSTM -> sigmoid, mirroring Ron's template structure."""
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(time_step, 1)),
        tf.keras.layers.LSTM(units, return_sequences=True, dropout=0.2,
                             recurrent_regularizer=tf.keras.regularizers.l2()),
        tf.keras.layers.LSTM(units, return_sequences=True, dropout=0.2,
                             recurrent_regularizer=tf.keras.regularizers.l2()),
        tf.keras.layers.LSTM(units, dropout=0.2,
                             recurrent_regularizer=tf.keras.regularizers.l2()),
        tf.keras.layers.Dense(1, activation='sigmoid'),
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model


def main():
    data = np.load(DATASET_NPZ, allow_pickle=True)
    X, y, groups = data['X'], data['y'], data['groups']
    time_step = X.shape[1]
    print("Loading dataset from:", DATASET_NPZ)
    print(f"Loaded {X.shape[0]} sequences from {len(np.unique(groups))} patients.")

    # ── PATIENT-LEVEL SPLIT ───────────────────────────────────────────────
    gss = GroupShuffleSplit(n_splits=1, test_size=TEST_FRAC, random_state=SEED)
    train_idx, test_idx = next(gss.split(X, y, groups))

    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    g_train, g_test = groups[train_idx], groups[test_idx]

    print(f"Train: {len(X_train)} sequences / {len(np.unique(g_train))} patients")
    print(f"Test : {len(X_test)} sequences / {len(np.unique(g_test))} patients")
    # Confirm no patient appears in both sides.
    overlap = set(g_train) & set(g_test)
    assert not overlap, f"LEAK: patients in both splits: {overlap}"
    print("Patient-level split verified: no overlap.\n")

    # ── NORMALIZE (fit on train only) ─────────────────────────────────────
    scaler = StandardScaler()
    n_tr, t, _ = X_train.shape
    scaler.fit(X_train.reshape(-1, 1))
    X_train = scaler.transform(X_train.reshape(-1, 1)).reshape(n_tr, t, 1)
    X_test  = scaler.transform(X_test.reshape(-1, 1)).reshape(len(X_test), t, 1)

    # ── TRAIN ─────────────────────────────────────────────────────────────
    model = build_model(time_step, UNITS)
    model.summary()

    # Class weighting: Controls (594) outnumber SKC (487). Without this the
    # model can minimize loss by just predicting the majority class, which is
    # exactly the "everything predicted Controls, val stuck at 44.7%" failure.
    # These weights make each SKC mistake cost proportionally more.
    n_total = len(y_train)
    n_ctrl  = int((y_train == 0).sum())
    n_skc   = int((y_train == 1).sum())
    class_weight = {0: n_total / (2 * n_ctrl), 1: n_total / (2 * n_skc)}
    print(f"\nClass weights (counteract imbalance): {class_weight}\n")

    # Hold out a few TRAIN patients for validation so we can watch overfitting
    # without touching the test set.
    gss_val = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)
    tr2_idx, val_idx = next(gss_val.split(X_train, y_train, g_train))
    history = model.fit(
        X_train[tr2_idx], y_train[tr2_idx],
        validation_data=(X_train[val_idx], y_train[val_idx]),
        epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=2,
        class_weight=class_weight,
    )

    # ── EVALUATE: SEQUENCE LEVEL ──────────────────────────────────────────
    probs = model.predict(X_test, verbose=0).ravel()
    seq_pred = (probs >= 0.5).astype(int)
    seq_acc = accuracy_score(y_test, seq_pred)
    print("\n" + "=" * 60)
    print(f"SEQUENCE-LEVEL test accuracy: {seq_acc*100:.1f}%  ({len(y_test)} sequences)")

    # ── EVALUATE: PATIENT LEVEL (the number to compare against ML) ─────────
    # Average the point predictions for each test patient, then threshold.
    print("\nPATIENT-LEVEL (majority vote of that patient's points):")
    pat_true, pat_pred = [], []
    for pid in np.unique(g_test):
        mask = g_test == pid
        true_label = int(round(y_test[mask].mean()))   # all same within patient
        vote = int(round(seq_pred[mask].mean()))        # fraction of points called SKC
        pat_true.append(true_label)
        pat_pred.append(vote)
        name = {0: 'Controls', 1: 'SKC'}
        mark = 'OK ' if vote == true_label else 'XX '
        print(f"  {mark}{pid:15s} true={name[true_label]:9s} pred={name[vote]:9s} "
              f"({seq_pred[mask].mean()*100:.0f}% of points called SKC)")

    pat_acc = accuracy_score(pat_true, pat_pred)
    print(f"\nPATIENT-LEVEL test accuracy: {pat_acc*100:.1f}%  ({len(pat_true)} patients)")
    print(f"Confusion matrix [rows=true Controls/SKC, cols=pred]:\n{confusion_matrix(pat_true, pat_pred)}")
    print("=" * 60)
    print("\nNOTE: a single 80/20 split on ~6 test patients is a very noisy")
    print("estimate. For a real number, wrap this in a leave-one-patient-out")
    print("or repeated GroupShuffleSplit loop once the model is behaving.")


if __name__ == '__main__':
    main()