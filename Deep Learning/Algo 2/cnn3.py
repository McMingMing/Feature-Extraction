"""
Brillouin Cornea DL - 1D CNN Classifier v3
==========================================
CHANGES FROM v1:
  - v1 used a single 80/20 patient split, which gave a very noisy estimate
    (one unlucky draw of 6 test patients could swing accuracy wildly).
  - v1 hardcoded a 0.5 sigmoid threshold, which failed when the model's
    outputs were systematically biased low.
  - v1 Dense layer was 64 units, which contributed to a 25.7% train-val gap.

CHANGES FROM v2:
  - v2 added repeated GroupShuffleSplit (10 splits), threshold tuning on the
    validation set, reduced Dense to 32 units, and an output-distribution
    diagnostic. Result: 65% avg patient accuracy but UNSTABLE. Of 10 splits,
    7 learned real separation, 2 were noise, 1 was inverted. Root cause: with
    only 24 training patients per split, the model sometimes converges to a
    bad local minimum and learns frame-position artifacts instead of the
    generalizable transition-zone shape.

  - v3 ADDS TEMPORAL JITTER AUGMENTATION. The discriminating signal lives in
    the SHAPE of the cornea-to-aqueous transition (frames ~55-65), but the
    exact frame where that transition sits varies between patients (scans
    start at slightly different depths). By randomly shifting each training
    sequence by a few frames each epoch, the model is forced to learn the
    transition SHAPE regardless of its absolute position, rather than
    memorizing "frame 60 = boundary" from the specific training patients.
    This directly targets the instability: the noise/inverted splits in v2
    were most likely the model overfitting to position artifacts.

  - v3 also adds EarlyStopping to prevent the overfitting seen in later epochs
    (v2 val_loss climbed steadily after ~epoch 10 while train kept improving).

WHAT TO WATCH:
  - If the 2 noise / 1 inverted splits become GOOD separation, the jitter
    fixed the position-overfitting problem.
  - If instability persists, the ceiling is the per-point SNR (1.48), not the
    architecture, and augmentation cannot fix that — the signal genuinely
    isn't separable enough at the single-sequence level.
"""

import numpy as np
import tensorflow as tf
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# ── CONFIGURATION ─────────────────────────────────────────────────────────
DATASET_NPZ = '/Users/minhnguyen/deep-learning/dl_dataset_2mm.npz'
EPOCHS      = 50          # higher ceiling; EarlyStopping will cut it short
BATCH_SIZE  = 16
TEST_FRAC   = 0.2
N_SPLITS    = 10
MAX_SHIFT   = 5           # max frames to jitter each sequence (+/-)
# ──────────────────────────────────────────────────────────────────────────


def jitter_sequences(X, max_shift, rng):
    """
    Randomly shift each sequence along the time axis by up to +/- max_shift
    frames. Vacated positions are filled by edge padding (repeat the end value)
    so we don't introduce artificial zeros. This teaches the model that the
    transition SHAPE matters, not its absolute frame index.
    """
    X_aug = np.empty_like(X)
    for i in range(X.shape[0]):
        shift = rng.integers(-max_shift, max_shift + 1)
        seq = X[i, :, 0]
        if shift == 0:
            X_aug[i, :, 0] = seq
        elif shift > 0:
            # shift right: pad front with first value
            X_aug[i, :, 0] = np.concatenate([np.full(shift, seq[0]), seq[:-shift]])
        else:
            # shift left: pad end with last value
            s = -shift
            X_aug[i, :, 0] = np.concatenate([seq[s:], np.full(s, seq[-1])])
    return X_aug


def build_model(time_step):
    """Shallow 1D CNN. Dense=32 (from v2). Structure unchanged from v2 so the
    only variable being tested is the augmentation."""
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(time_step, 1)),
        tf.keras.layers.Conv1D(32, kernel_size=7, activation='relu', padding='same'),
        tf.keras.layers.MaxPooling1D(pool_size=2),
        tf.keras.layers.Conv1D(64, kernel_size=5, activation='relu', padding='same'),
        tf.keras.layers.MaxPooling1D(pool_size=2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(1, activation='sigmoid'),
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model


def find_best_threshold(probs, y_true):
    best_thresh, best_acc = 0.5, 0.0
    for t in np.arange(0.2, 0.81, 0.05):
        acc = accuracy_score(y_true, (probs >= t).astype(int))
        if acc > best_acc:
            best_acc = acc
            best_thresh = t
    return best_thresh


def diagnose_outputs(probs, y_test, split_i):
    """Print raw sigmoid output distributions for Controls vs SKC."""
    ctrl_probs = probs[y_test == 0]
    skc_probs  = probs[y_test == 1]
    sep = skc_probs.mean() - ctrl_probs.mean()

    print(f"  Output distribution (split {split_i+1}):")
    print(f"    Controls: mean={ctrl_probs.mean():.3f}  std={ctrl_probs.std():.3f}  "
          f"range=[{ctrl_probs.min():.3f}, {ctrl_probs.max():.3f}]  n={len(ctrl_probs)}")
    print(f"    SKC:      mean={skc_probs.mean():.3f}  std={skc_probs.std():.3f}  "
          f"range=[{skc_probs.min():.3f}, {skc_probs.max():.3f}]  n={len(skc_probs)}")
    print(f"    Separation (SKC - Controls): {sep:+.3f}", end="  ")

    if abs(sep) < 0.05:
        print(">> NOISE: near-zero separation, model not learning.")
    elif sep < 0:
        print(">> INVERTED: SKC lower than Controls, labels may be flipped.")
    elif sep < 0.10:
        print(">> WEAK: small but positive separation.")
    else:
        print(">> GOOD: meaningful positive separation.")


def main():
    print(f"Loading dataset from: {DATASET_NPZ}")
    data = np.load(DATASET_NPZ, allow_pickle=True)
    X, y, groups = data['X'], data['y'], data['groups']
    time_step = X.shape[1]

    print(f"Loaded {X.shape[0]} sequences from {len(np.unique(groups))} patients.")
    print(f"Class balance: Controls={int((y==0).sum())}  SKC={int((y==1).sum())}")
    print(f"Augmentation: temporal jitter +/-{MAX_SHIFT} frames\n")

    all_seq_accs, all_pat_accs = [], []
    gss = GroupShuffleSplit(n_splits=N_SPLITS, test_size=TEST_FRAC, random_state=42)

    for split_i, (train_idx, test_idx) in enumerate(gss.split(X, y, groups)):
        tf.random.set_seed(split_i)
        np.random.seed(split_i)
        rng = np.random.default_rng(split_i)

        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        g_train, g_test = groups[train_idx], groups[test_idx]

        # Normalize fit on train only
        scaler = StandardScaler()
        n_tr, t, _ = X_train.shape
        scaler.fit(X_train.reshape(-1, 1))
        X_train_s = scaler.transform(X_train.reshape(-1, 1)).reshape(n_tr, t, 1)
        X_test_s  = scaler.transform(X_test.reshape(-1, 1)).reshape(len(X_test), t, 1)

        # Class weights
        n_total = len(y_train)
        n_ctrl  = int((y_train == 0).sum())
        n_skc   = int((y_train == 1).sum())
        class_weight = {0: n_total / (2 * n_ctrl), 1: n_total / (2 * n_skc)}

        # Validation split from train
        gss_val = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=split_i)
        tr2_idx, val_idx = next(gss_val.split(X_train_s, y_train, g_train))

        # AUGMENT the training portion only (never validation or test)
        X_tr2 = jitter_sequences(X_train_s[tr2_idx], MAX_SHIFT, rng)
        y_tr2 = y_train[tr2_idx]

        # Train with EarlyStopping
        model = build_model(time_step)
        es = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=8, restore_best_weights=True
        )
        model.fit(
            X_tr2, y_tr2,
            validation_data=(X_train_s[val_idx], y_train[val_idx]),
            epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=0,
            class_weight=class_weight, callbacks=[es],
        )

        # Threshold on validation set
        val_probs   = model.predict(X_train_s[val_idx], verbose=0).ravel()
        best_thresh = find_best_threshold(val_probs, y_train[val_idx])

        # Evaluate on test
        test_probs = model.predict(X_test_s, verbose=0).ravel()
        diagnose_outputs(test_probs, y_test, split_i)

        seq_pred = (test_probs >= best_thresh).astype(int)
        seq_acc  = accuracy_score(y_test, seq_pred)
        all_seq_accs.append(seq_acc)

        pat_true, pat_pred = [], []
        for pid in np.unique(g_test):
            mask = g_test == pid
            pat_true.append(int(round(y_test[mask].mean())))
            pat_pred.append(int(round(seq_pred[mask].mean())))
        pat_acc = accuracy_score(pat_true, pat_pred)
        all_pat_accs.append(pat_acc)

        print(f"  Result: seq={seq_acc*100:5.1f}%  patient={pat_acc*100:5.1f}%  "
              f"threshold={best_thresh:.2f}\n")

    print("=" * 60)
    print(f"AVERAGED OVER {N_SPLITS} SPLITS (v3, with jitter augmentation):")
    print(f"  Sequence-level accuracy: {np.mean(all_seq_accs)*100:.1f}% "
          f"(std {np.std(all_seq_accs)*100:.1f}%)")
    print(f"  Patient-level accuracy:  {np.mean(all_pat_accs)*100:.1f}% "
          f"(std {np.std(all_pat_accs)*100:.1f}%)")
    print("=" * 60)
    print("\nCompare against v2 (no augmentation): 65.0% patient (std 15.7%).")
    print("If std dropped, jitter stabilized training. If not, the ceiling is")
    print("the per-point SNR (1.48), which augmentation cannot fix.")


if __name__ == '__main__':
    main()