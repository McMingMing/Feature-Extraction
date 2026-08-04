"""
Brillouin Cornea Segmentation - 1D U-Net
=========================================
Trains a 1D U-Net to segment each 100-frame depth profile into cornea (1) vs
not-cornea (0), following the TensorFlow segmentation tutorial Ron linked but
adapted from 2D images to 1D signals.

Pipeline: depth profile -> U-Net -> predicted cornea mask -> mean over cornea
frames -> compare that mean between Controls and SKC (the box plot Ron wants).

Ron said we can use ALL the data here since the goal is to classify cornea, not
diagnose per point. We still do a PATIENT-LEVEL split so segmentation quality
reflects unseen patients.

Architecture (1D U-Net, 2 pooling levels):
  100 -> 50 -> 25 (bottleneck) -> 50 -> 100, skip connections line up cleanly.
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ── CONFIGURATION ─────────────────────────────────────────────────────────
DATASET_NPZ  = '/Users/minhnguyen/deep-learning/labels_dataset.npz'
PLOT_DIR     = '/Users/minhnguyen/deep-learning/segmentation_plots'
EPOCHS       = 40
BATCH_SIZE   = 16
TEST_FRAC    = 0.2
SEED         = 1
N_PRED_PLOTS = 8
# Spatial restriction. Ron: "Do all the points versus 2mm restricted. The idea
# would be to prove that the disease impacts the center of the eye more than the
# outer ring." Set RESTRICT_2MM = True to train/evaluate on central points only.
RESTRICT_2MM = False
COORD_FILE   = '/Users/minhnguyen/deep-learning/combined_data.xlsx'
RADIUS_MM    = 2.0

# --- FSR / true Brillouin shift conversion (Ron's MATLAB) ---
#   FSR = 2*const + water;  true_shift = (FSR - plotted)/2
# NOTE: this transform is DECREASING in the plotted value, so the MIN true
# shift corresponds to the MAX plotted value. Always convert to true shift
# BEFORE taking min/max, or the two get silently swapped.
FB_WATER  = 5.07790995626263
FSR       = 2 * FB_WATER + FB_WATER      # 15.2337 GHz

# --- Reference values from Zhang et al., AJO 2023 (Table 4), Controls vs KC.
# Their pipeline used manually-selected plateau regions; ours is automated, so
# these are the benchmark our metrics are being compared against.
PAPER_REF = {
    'Mean':    (5.713, 5.679, 0.0004,   0.96),
    'Max':     (5.746, 5.724, 0.089,    None),
    'Min':     (5.696, 5.641, 0.000002, 1.00),
    'SSD':     (0.012, 0.018, 0.03,     0.82),
    'Max-Min': (0.049, 0.083, 0.0047,   0.87),
}
# Robust min percentile. The paper's absolute Min is an extreme-value statistic;
# our automated per-point estimates are noisier than their manual ones (spatial
# SD ~0.030 vs their 0.012), so the absolute min chases noise. Taking a low
# percentile instead recovers most of the signal.
ROBUST_MIN_PCT = 10

N_SPLITS     = 5   # repeat train/test with different patient splits and                    # average, since a single 6-patient test split is noisy
                    # (verified: one split gave Controls=SKC on predictions,
                    # even though the TRUE labels show a highly significant
                    # difference across the full dataset, p<0.0001)
# ──────────────────────────────────────────────────────────────────────────

tf.random.set_seed(SEED)
np.random.seed(SEED)


def apply_2mm_filter(X, Y_seg, Y_cls, groups, coord_file, radius_mm):
    """
    Keep only sequences whose spatial point lies within radius_mm of the cornea
    center. Column order in each patient's .mat matches row order in
    combined_data.xlsx for that patient, so the Nth sequence for a patient
    corresponds to that patient's Nth row.
    """
    import pandas as pd
    df = pd.read_excel(coord_file)
    df['Patient'] = df['Patient'].astype(str)
    df['radius'] = np.sqrt(df['X (mm)']**2 + df['Y (mm)']**2)
    radius_by_patient = {pid: g['radius'].to_numpy()
                         for pid, g in df.groupby('Patient')}

    keep = np.zeros(len(X), dtype=bool)
    seen = {}
    for i, pid in enumerate(groups):
        k = seen.get(pid, 0); seen[pid] = k + 1
        r = radius_by_patient.get(pid)
        if r is None:
            r = radius_by_patient.get(pid.split()[0])
        if r is not None and k < len(r) and r[k] <= radius_mm:
            keep[i] = True
    return X[keep], Y_seg[keep], Y_cls[keep], groups[keep]


def to_true_shift(plotted_ghz):
    """Plotted value -> true Brillouin shift. DECREASING transform."""
    return (FSR - plotted_ghz) / 2.0


def paper_metrics(point_values_true):
    """
    Compute the five metrics from Zhang et al. Table 4 for one patient, given
    that patient's per-point plateau values already in TRUE shift (GHz).
    Plus a noise-robust variant of Min.
    """
    v = np.asarray(point_values_true)
    return {
        'Mean':    v.mean(),
        'Max':     v.max(),                              # stiffest point
        'Min':     v.min(),                              # softest point (the KC cone)
        'Min_p10': np.percentile(v, ROBUST_MIN_PCT),     # noise-robust softest
        'SSD':     v.std(),                              # spatial variability
        'Max-Min': v.max() - v.min(),
    }


def dice_coef(y_true, y_pred, smooth=1.0):
    """Overlap metric. 1.0 = perfect overlap. More informative than accuracy
    when cornea/non-cornea frames are imbalanced."""
    y_true_f = tf.reshape(y_true, [-1])
    y_pred_f = tf.reshape(tf.cast(y_pred > 0.5, tf.float32), [-1])
    inter = tf.reduce_sum(y_true_f * y_pred_f)
    return (2. * inter + smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)


def build_unet_1d(time_step=100):
    inp = layers.Input(shape=(time_step, 1))

    c1 = layers.Conv1D(32, 5, activation='relu', padding='same')(inp)
    c1 = layers.Conv1D(32, 5, activation='relu', padding='same')(c1)
    p1 = layers.MaxPooling1D(2)(c1)                       # 100 -> 50

    c2 = layers.Conv1D(64, 5, activation='relu', padding='same')(p1)
    c2 = layers.Conv1D(64, 5, activation='relu', padding='same')(c2)
    p2 = layers.MaxPooling1D(2)(c2)                       # 50 -> 25

    b = layers.Conv1D(128, 5, activation='relu', padding='same')(p2)
    b = layers.Conv1D(128, 5, activation='relu', padding='same')(b)
    b = layers.Dropout(0.3)(b)

    u2 = layers.UpSampling1D(2)(b)                        # 25 -> 50
    u2 = layers.Concatenate()([u2, c2])
    c3 = layers.Conv1D(64, 5, activation='relu', padding='same')(u2)
    c3 = layers.Conv1D(64, 5, activation='relu', padding='same')(c3)

    u1 = layers.UpSampling1D(2)(c3)                       # 50 -> 100
    u1 = layers.Concatenate()([u1, c1])
    c4 = layers.Conv1D(32, 5, activation='relu', padding='same')(u1)
    c4 = layers.Conv1D(32, 5, activation='relu', padding='same')(c4)

    out = layers.Conv1D(1, 1, activation='sigmoid')(c4)  # (100,1) per-frame prob

    model = tf.keras.Model(inp, out)
    model.compile(optimizer='adam', loss='binary_crossentropy',
                  metrics=['accuracy', dice_coef])
    return model


def plot_prediction(seq, true_mask, pred_mask, idx, save_path):
    frames = np.arange(len(seq))
    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor('#16213e'); ax.set_facecolor('#1a1a2e')
    ax.plot(frames, seq, '-', color='lime', linewidth=1.5, alpha=0.7, label='Signal (normalized)')
    tf_ = frames[true_mask == 1]; pf_ = frames[pred_mask == 1]
    if len(tf_):
        ax.scatter(tf_, seq[tf_], color='cyan', s=60, alpha=0.4, label=f'True cornea (n={len(tf_)})')
    if len(pf_):
        ax.scatter(pf_, seq[pf_], color='yellow', s=20, zorder=3, label=f'Predicted cornea (n={len(pf_)})')
    ax.set_title(f'Segmentation prediction — test sequence {idx}', color='white', fontsize=12, fontweight='bold')
    ax.set_xlabel('Frame (depth)', color='white'); ax.set_ylabel('Normalized shift', color='white')
    ax.tick_params(colors='white')
    ax.legend(fontsize=9, facecolor='#0f172a', labelcolor='white'); ax.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.savefig(save_path, dpi=120, facecolor=fig.get_facecolor(), bbox_inches='tight')
    plt.close()


def main():
    os.makedirs(PLOT_DIR, exist_ok=True)

    print(f"Loading dataset from: {DATASET_NPZ}")
    data = np.load(DATASET_NPZ, allow_pickle=True)
    X, Y_seg, Y_cls, groups = data['X'], data['Y_seg'], data['Y_cls'], data['groups']
    time_step = X.shape[1]
    Y_seg = Y_seg[..., np.newaxis].astype(np.float32)     # (n,100,1)

    print(f"Loaded {X.shape[0]} sequences from {len(np.unique(groups))} patients.")

    if RESTRICT_2MM:
        n0 = len(X)
        X, Y_seg, Y_cls, groups = apply_2mm_filter(
            X, Y_seg, Y_cls, groups, COORD_FILE, RADIUS_MM)
        print(f"2mm restriction ON: {n0} -> {len(X)} sequences "
              f"({len(np.unique(groups))} patients)")
    else:
        print("2mm restriction OFF: using all spatial points")

    print(f"X: {X.shape}   Y_seg: {Y_seg.shape}")
    print(f"Avg cornea frames per sequence: {Y_seg.mean()*100:.1f}\n")

    gss = GroupShuffleSplit(n_splits=N_SPLITS, test_size=TEST_FRAC, random_state=SEED)

    all_ctrl, all_skc = [], []          # point-level (what Ron saw)
    pat_ctrl, pat_skc = [], []          # patient-level (Ron: 'having individual
                                        # points and comparing it to Hongyuan
                                        # value would give a better picture')
    patient_rows = []                   # (patient, diagnosis, mean, n_points)
    frame_accs, dices = [], []
    last_split_data = None   # keep the last split's test set for example plots

    for split_i, (tr_idx, te_idx) in enumerate(gss.split(X, Y_cls, groups)):
        tf.random.set_seed(split_i)
        np.random.seed(split_i)

        X_tr, X_te = X[tr_idx], X[te_idx]
        Yseg_tr, Yseg_te = Y_seg[tr_idx], Y_seg[te_idx]
        Ycls_te = Y_cls[te_idx]
        g_tr, g_te = groups[tr_idx], groups[te_idx]
        assert not (set(g_tr) & set(g_te)), "LEAK: patient in both splits"

        scaler = StandardScaler()
        n_tr = len(X_tr)
        scaler.fit(X_tr.reshape(-1, 1))
        X_tr_s = scaler.transform(X_tr.reshape(-1, 1)).reshape(n_tr, time_step, 1)
        X_te_s = scaler.transform(X_te.reshape(-1, 1)).reshape(len(X_te), time_step, 1)

        model = build_unet_1d(time_step)
        es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=8,
                                              restore_best_weights=True)
        model.fit(X_tr_s, Yseg_tr, validation_split=0.2, epochs=EPOCHS,
                  batch_size=BATCH_SIZE, verbose=0, callbacks=[es])

        # Use direct call instead of .predict() — .predict() builds a new
        # tf.function per model instance, which triggers retracing warnings and
        # wasted recompilation inside this multi-split loop.
        probs = model(X_te_s, training=False).numpy()
        pred_mask = (probs > 0.5).astype(np.int8)

        frame_acc = (pred_mask == Yseg_te.astype(np.int8)).mean()
        d = dice_coef(Yseg_te, probs).numpy()
        frame_accs.append(frame_acc); dices.append(float(d))

        corn_means = []
        for i in range(len(X_te)):
            m = pred_mask[i, :, 0] == 1
            corn_means.append(X_te[i][m, 0].mean() if m.any() else np.nan)
        corn_means = np.array(corn_means)

        ctrl = corn_means[(Ycls_te == 0) & ~np.isnan(corn_means)] / 1e9
        skc  = corn_means[(Ycls_te == 1) & ~np.isnan(corn_means)] / 1e9
        all_ctrl.append(ctrl); all_skc.append(skc)

        # Per-patient aggregation. Convert each point to TRUE shift first, then
        # compute the paper's spatial metrics across that patient's map.
        for pid in np.unique(g_te):
            m = (g_te == pid) & ~np.isnan(corn_means)
            if not m.any():
                continue
            pts_true = to_true_shift(corn_means[m] / 1e9)   # true shift per point
            mets = paper_metrics(pts_true)
            diag = int(Ycls_te[g_te == pid][0])
            patient_rows.append({
                'patient': pid,
                'diagnosis': 'SKC' if diag else 'Controls',
                'label': diag,
                'n_points': int(m.sum()),
                'split': split_i + 1,
                **mets,
            })
            (pat_skc if diag else pat_ctrl).append(mets['Mean'])

        print(f"Split {split_i+1}/{N_SPLITS}: frame_acc={frame_acc*100:.1f}%  "
              f"dice={d:.3f}  Controls={ctrl.mean():.3f}  SKC={skc.mean():.3f}  "
              f"({len(np.unique(g_te))} test patients)")

        if split_i == N_SPLITS - 1:
            last_split_data = (X_te_s, Yseg_te, pred_mask)

    ctrl_all = np.concatenate(all_ctrl)
    skc_all  = np.concatenate(all_skc)

    print("\n" + "=" * 60)
    print(f"AVERAGED OVER {N_SPLITS} SPLITS:")
    print(f"  Per-frame accuracy: {np.mean(frame_accs)*100:.1f}% "
          f"(std {np.std(frame_accs)*100:.1f}%)")
    print(f"  Dice coefficient:   {np.mean(dices):.3f} (std {np.std(dices):.3f})")
    print(f"\nPREDICTED cornea mean, pooled across all {N_SPLITS} test sets "
          f"(plotted GHz; apply FSR for true shift):")
    print(f"  Controls: {ctrl_all.mean():.3f} +/- {ctrl_all.std():.3f}  (n={len(ctrl_all)})")
    print(f"  SKC:      {skc_all.mean():.3f} +/- {skc_all.std():.3f}  (n={len(skc_all)})")
    t, p = stats.ttest_ind(ctrl_all, skc_all)
    print(f"  Difference: {ctrl_all.mean()-skc_all.mean():+.4f} GHz   t-test p={p:.4f}")
    print("=" * 60)

    # ── PAPER METRIC COMPARISON (Zhang et al. AJO 2023, Table 4) ──────────
    # Each patient contributes ONE value per metric, computed across that
    # patient's spatial map of predicted cornea plateaus (in TRUE shift).
    from sklearn.metrics import roc_auc_score

    labels = np.array([r['label'] for r in patient_rows])
    metric_names = ['Mean', 'Max', 'Min', 'Min_p10', 'SSD', 'Max-Min']

    print(f"\nPER-PATIENT METRICS vs Zhang et al. AJO 2023 (Table 4)")
    print("Each patient = one value per metric, computed across its spatial map.")
    print("Values in TRUE Brillouin shift (GHz).\n")
    print(f"{'Metric':9s} {'Controls':>15s} {'SKC':>15s} {'p':>9s} {'AUC':>6s}   "
          f"{'paper p':>9s} {'paper AUC':>9s}")
    print("-" * 84)

    results = {}
    for nm in metric_names:
        v = np.array([r[nm] for r in patient_rows])
        c0, c1 = v[labels == 0], v[labels == 1]
        if len(c0) < 2 or len(c1) < 2:
            continue
        _, pv = stats.ttest_ind(c0, c1)
        auc = roc_auc_score(labels, v)
        auc = max(auc, 1 - auc)          # orientation-free discriminability
        results[nm] = (c0, c1, pv, auc)

        ref = PAPER_REF.get(nm)
        if ref:
            rp, ra = ref[2], ref[3]
            rp_s = f"{rp:.6f}" if rp < 0.001 else f"{rp:.4f}"
            ra_s = f"{ra:.2f}" if ra is not None else "--"
        else:
            rp_s, ra_s = "--", "--"     # Min_p10 has no paper counterpart

        print(f"{nm:9s} {c0.mean():7.3f}+/-{c0.std():.3f} {c1.mean():7.3f}+/-{c1.std():.3f} "
              f"{pv:9.5f} {auc:6.3f}   {rp_s:>9s} {ra_s:>9s}")

    best = max(results, key=lambda k: results[k][3])
    print(f"\nBest discriminator here: {best} (AUC {results[best][3]:.3f})")
    print("Paper's best was Min (AUC 1.00). If Min underperforms here, it is")
    print("because Min is an extreme-value statistic and our automated per-point")
    print("estimates are noisier than their manual ones -- one bad point drags")
    print("the minimum. Min_p10 is the noise-robust version of the same idea.")

    ssd_ctrl = results.get('SSD', (np.array([np.nan]),))[0].mean()
    print(f"\nSpatial SD (Controls): ours {ssd_ctrl:.4f} vs paper 0.012 "
          f"-> {ssd_ctrl/0.012:.1f}x their per-point noise")
    print("=" * 84)

    # Per-patient table -> CSV with every metric, ready to line up against
    # the manual values one-to-one.
    csv_path = os.path.join(PLOT_DIR, 'per_patient_metrics.csv')
    with open(csv_path, 'w') as f:
        f.write('patient,diagnosis,split,n_points,' + ','.join(metric_names) + '\n')
        for r in sorted(patient_rows, key=lambda x: (x['patient'], x['split'])):
            vals = ','.join(f"{r[nm]:.6f}" for nm in metric_names)
            f.write(f"{r['patient']},{r['diagnosis']},{r['split']},"
                    f"{r['n_points']},{vals}\n")
    print(f"\nPer-patient metrics saved: {csv_path}")

    pat_ctrl_a = np.array(pat_ctrl); pat_skc_a = np.array(pat_skc)

    # ── BOX PLOTS: one panel per metric, mirroring the paper's Figure 3 ────
    plot_metrics = ['Mean', 'Max', 'Min', 'Min_p10', 'SSD', 'Max-Min']
    fig, axes = plt.subplots(2, 3, figsize=(14, 9))
    for ax, nm in zip(axes.ravel(), plot_metrics):
        if nm not in results:
            ax.axis('off'); continue
        c0, c1, pv, auc = results[nm]
        ax.boxplot([c0, c1], tick_labels=['Controls', 'SKC'])
        ax.set_title(f'{nm}\np={pv:.4f}   AUC={auc:.3f}', fontsize=11)
        ax.set_ylabel('True Brillouin shift (GHz)' if nm not in ('SSD', 'Max-Min')
                      else 'GHz')
        ax.grid(True, alpha=0.3)

        # Overlay the paper's reported values as dashed reference lines
        ref = PAPER_REF.get(nm)
        if ref:
            ax.axhline(ref[0], color='tab:blue', ls='--', lw=1, alpha=0.6)
            ax.axhline(ref[1], color='tab:red',  ls='--', lw=1, alpha=0.6)

    suffix = '2mm restricted' if RESTRICT_2MM else 'all points'
    fig.suptitle(f'Per-patient Brillouin metrics from U-Net segmentation ({suffix})\n'
                 f'dashed lines = Zhang et al. AJO 2023 reported values '
                 f'(blue=Controls, red=KC)', fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, 'paper_metrics_boxplot.png'),
                dpi=120, bbox_inches='tight')
    plt.close()

    # Point-level vs patient-level, showing why the spread narrows
    fig, axes = plt.subplots(1, 2, figsize=(11, 6))
    axes[0].boxplot([to_true_shift(ctrl_all), to_true_shift(skc_all)],
                    tick_labels=['Controls', 'SKC'])
    axes[0].set_ylabel('True Brillouin shift (GHz)')
    axes[0].set_title(f'Point level\n(every point pooled, n={len(ctrl_all)+len(skc_all)})')
    axes[0].grid(True, alpha=0.3)

    axes[1].boxplot([pat_ctrl_a, pat_skc_a], tick_labels=['Controls', 'SKC'])
    axes[1].set_title(f'Patient level\n(one value per patient, n={len(pat_ctrl_a)+len(pat_skc_a)})')
    axes[1].grid(True, alpha=0.3)

    ylim = (min(axes[0].get_ylim()[0], axes[1].get_ylim()[0]),
            max(axes[0].get_ylim()[1], axes[1].get_ylim()[1]))
    for a in axes:
        a.set_ylim(ylim)

    fig.suptitle(f'Cornea mean stiffness ({suffix})')
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, 'cornea_mean_boxplot.png'), dpi=120, bbox_inches='tight')
    plt.close()

    X_te_s, Yseg_te, pred_mask = last_split_data
    for i in range(min(N_PRED_PLOTS, len(X_te_s))):
        plot_prediction(X_te_s[i, :, 0], Yseg_te[i, :, 0].astype(int),
                        pred_mask[i, :, 0], i,
                        os.path.join(PLOT_DIR, f'prediction_{i}.png'))
    print(f"\nSaved prediction plots + box plot to {PLOT_DIR}")
    print("Send these to Ron to verify the model's cornea detection looks right.")


if __name__ == '__main__':
    main()