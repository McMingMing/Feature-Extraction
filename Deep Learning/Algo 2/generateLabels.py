"""
Brillouin Cornea Segmentation - Label Generator
================================================
Converts each spatial point's depth profile from shifts_before.mat into
a binary label array of length 100:
  1 = cornea frame (Brillouin shift is in the corneal plateau region)
  0 = everything else (aqueous, noise, pre-cornea)

This is the ground truth for the segmentation model Ron described:
  "it'll look something like (0,0,0,0,0,1,1,1,1,1,0,0,0,...) for all
   100 points where 1 is the points that are cornea and 0 is everything else."

HOW THE LABELING WORKS
-----------------------
For each column (spatial point) in the .mat array:
1. Smooth the 100-frame shift signal with a Gaussian filter.
2. Find the steepest negative slope (the cornea-to-aqueous boundary).
3. Use Otsu's method to find the natural threshold between cornea and aqueous.
4. Label frames BEFORE the boundary AND above the Otsu threshold as cornea (1).
5. Everything else is labeled 0.

This is the same logic as sCurveTesting.py, adapted to work directly on
the .mat shift values instead of raw TDMS cube data.

OUTPUT
------
- labels_dataset.npz: X (n,100,1), y_seg (n,100) binary labels, y_cls (n,) diagnosis, groups (n,)
- PNG plots of example profiles with labels overlaid (for Ron to verify)

IMPORTANT: Send Ron the example plots before training the segmentation model.
"""

import os
import glob
import numpy as np
import pandas as pd
import scipy.io
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d

# ── CONFIGURATION ─────────────────────────────────────────────────────────
DATA_DIR       = '/Users/minhnguyen/deep-learning/All Brillouin Point Data'
COORD_FILE     = '/Users/minhnguyen/deep-learning/combined_data.xlsx'
MAT_FILENAME   = 'shifts_before.mat'
MAT_KEY        = 'shifts_before'
OUTPUT_NPZ     = '/Users/minhnguyen/deep-learning/labels_dataset.npz'
PLOT_DIR       = '/Users/minhnguyen/deep-learning/label_verification_plots'

LABEL_MAP      = {'Controls': 0, 'SKC': 1}
N_EXAMPLE_PLOTS = 6   # how many example profiles to plot for Ron's verification
FLAT_FRAC       = 0.25  # frame is 'flat' if |slope| <= this fraction of max|slope|
MIN_FRAMES      = 10    # if flatness leaves fewer frames than this, relax it
QC_MAX_GHZ      = 4.0   # cornea mean ABOVE this = flagged (locked onto aqueous)
QC_SLOPE_MULT   = 4.0   # flag if mean |slope| in the labeled region exceeds this
                        # multiple of the dataset-wide median slope. Ron spotted
                        # one of these by eye ("in prediction_7 the true cornea
                        # dips a bit too low"); that sequence had ~6x the median
                        # slope, i.e. the label sat on the transition, not the
                        # plateau. Affects ~3.7% of sequences.

# --- Cornea START detection (Ron: "There is some air in front of the cornea
# that gets acquired... your code doesn't currently have a good idea where it
# starts. I would add the code of just figuring out where the cornea starts
# from the raw values.") ---
START_WIN     = 7     # rolling window (frames) for measuring raw noise
START_MULT    = 1.8   # frame is "quiet" if its noise <= MULT * median noise
START_RUN     = 6     # need this many consecutive quiet frames to call the start
START_TOL     = 1.0   # fraction of the run that must be quiet (1.0 = all)

# --- FSR / true Brillouin shift conversion (from Ron's MATLAB screenshot) ---
#   FSRBefore = 2*5.07790995626263 + water_before
#   shiftsbefore_abs = (FSRBefore - shifts_before)/2
# NOTE: Ron's prose said "2*(Constant)*Water" but the MATLAB is 2*Constant +
# water (addition). His earlier email said "FSR - 2*shift" but the MATLAB is
# (FSR - shift)/2. The MATLAB is authoritative: it reproduces his stated
# cornea ~5.7 / aqueous ~5.2 GHz, and gives FSR ~15.23 ("around 15 or so").
FB_WATER_CONST = 5.07790995626263
WATER_BEFORE   = 5.07790995626263   # from fb_water.mat
FSR            = 2 * FB_WATER_CONST + WATER_BEFORE
EXCLUDE_FLAGGED = True  # Ron: "For the values that are flagged I wouldn't use
                        #       them for now, since we just want to use the good
                        #       ones for training."
# ──────────────────────────────────────────────────────────────────────────


def get_otsu_threshold(data):
    """Otsu's method for 1D data. Finds the natural split between two groups."""
    if len(data) < 2:
        return np.mean(data) if len(data) else 0
    sorted_data = np.sort(data)
    thresholds = (sorted_data[:-1] + sorted_data[1:]) / 2.0
    best_var, best_t = -1, np.mean(data)
    for t in thresholds:
        c1 = data[data >= t]; c2 = data[data < t]
        if len(c1) == 0 or len(c2) == 0: continue
        w1 = len(c1)/len(data); w2 = len(c2)/len(data)
        var = w1 * w2 * (np.mean(c1) - np.mean(c2))**2
        if var > best_var:
            best_var = var; best_t = t
    return best_t


def to_true_shift(plotted_ghz):
    """Convert plotted value -> true Brillouin shift, per Ron's MATLAB.
    Note this transform is DECREASING: a LOWER plotted value maps to a HIGHER
    true shift. That is why the cornea (true ~5.7) is the LOW plotted region."""
    return (FSR - plotted_ghz) / 2.0


def find_cornea_start(raw_ghz, win=START_WIN, mult=START_MULT,
                      run=START_RUN, tol=START_TOL):
    """
    Find where the cornea starts, using the RAW (unsmoothed) values.

    Ron: "All, if not most of all the points don't actually start at the
    cornea. There is some air in front of the cornea that gets acquired."
    and "the smoothing filtering masks the noise in the beginning."

    Air has no coherent Brillouin signal, so the raw trace is erratic there.
    We DETREND first (subtract a light smooth) so the S-curve's own slope is
    not mistaken for noise, then measure rolling noise and return the first
    frame that begins a sustained quiet run.
    """
    sm = gaussian_filter1d(raw_ghz, 2)
    resid = raw_ghz - sm
    half = win // 2
    rstd = np.array([
        np.std(resid[max(0, i - half):min(len(resid), i + half + 1)])
        for i in range(len(resid))
    ])
    med = np.median(rstd)
    if med <= 0:
        return 0
    quiet = rstd <= (mult * med)
    for i in range(len(quiet) - run):
        if quiet[i:i + run].mean() >= tol:
            return i
    return 0


def largest_block(mask):
    """Keep only the longest contiguous run of True in a boolean mask.

    Some profiles have low values in more than one place. The real cornea
    plateau is the longer, sustained block; short runs elsewhere are noise.
    """
    best_s, best_e, best_len = 0, 0, 0
    s = None
    for i, v in enumerate(mask):
        if v and s is None:
            s = i
        elif not v and s is not None:
            if i - s > best_len:
                best_s, best_e, best_len = s, i, i - s
            s = None
    if s is not None and len(mask) - s > best_len:
        best_s, best_e, best_len = s, len(mask), len(mask) - s
    out = np.zeros_like(mask)
    out[best_s:best_e] = True
    return out


def generate_label(seq, sigma=3, flat_frac=FLAT_FRAC, min_frames=MIN_FRAMES):
    """
    Generate binary cornea label for one 100-frame depth profile.
    1 = cornea (the flat plateau), 0 = everything else.

    *** ORIENTATION: CORNEA IS THE LOW-VALUE REGION ***
    Ron (round 2): "The sequences you have are actually flipped. The cornea is
    actually the bottom left and the super flat region that you got is the
    aqueous humor." So we label the LOW flat region, not the high one.

    Why the plotted values differ from Ron's quoted GHz: the true Brillouin
    shift is FSR - 2 * plotted_value. Because that transform is DECREASING in
    the plotted value, the LOWER plotted value maps to the HIGHER true shift.
    Cornea (~5.7) > aqueous (~5.2) in true shift, so cornea is the LOWER
    plotted value. This is consistent with Ron's orientation call.
    (Caveat: Ron's worked example used FSR=15, which maps the HIGH plotted
    value to 5.7 instead. The exact FSR is unresolved pending water.mat.
    Since the transform is monotonic and applied uniformly, it does not change
    the SHAPE the model learns, so labeling can proceed now and the true GHz
    conversion can be applied later for the final box plot.)

    THREE CRITERIA:
    1. LOW: value <= Otsu threshold. Otsu finds the natural high/low split in
       whatever scale the data is in, so it works without knowing the FSR.
    2. FLAT: |slope| <= flat_frac * max|slope|. Ron: "the flat part of your
       smoothed out curve is what you should aim at (try not to include
       points that are the slope)".
    3. LARGEST CONTIGUOUS BLOCK: resolves the double-ended profiles Ron
       flagged as noise.

    Guard: if the flatness filter leaves fewer than min_frames, relax it and
    take the largest low block instead, so a noisy profile still gets a
    usable label rather than 2-3 stray frames.

    Returns:
        label (np.array, shape 100): 1=cornea plateau (low region), 0=other
        boundary_frame (int): first cornea frame (for plotting)
        otsu_thresh (float): the high/low threshold used
    """
    smoothed = gaussian_filter1d(seq.astype(np.float64), sigma=sigma)
    otsu_thresh = get_otsu_threshold(smoothed)

    low = smoothed <= otsu_thresh          # cornea = LOW region (flipped)
    slope = np.abs(np.gradient(smoothed))
    flat = slope <= (flat_frac * slope.max()) if slope.max() > 0 else np.ones_like(low)

    # Cornea START from the RAW values: everything before this is air/noise.
    start = find_cornea_start(seq.astype(np.float64) / 1e9)
    after_start = np.zeros(len(seq), dtype=bool)
    after_start[start:] = True

    mask = largest_block(low & flat & after_start)
    if mask.sum() < min_frames:
        mask = largest_block(low & after_start)
    if mask.sum() < min_frames:
        mask = largest_block(low)

    label = mask.astype(np.int8)
    cornea_frames = np.where(label == 1)[0]
    boundary_frame = int(cornea_frames[0]) if len(cornea_frames) else 50
    return label, boundary_frame, otsu_thresh, start


def plot_example(seq, label, boundary, otsu_thresh, patient_id, point_idx,
                 diagnosis, save_path, start=0):
    """Plot one depth profile with its cornea labels overlaid."""
    frames = np.arange(len(seq))
    smoothed = gaussian_filter1d(seq.astype(np.float64), sigma=3)

    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor('#16213e')
    ax.set_facecolor('#1a1a2e')

    ax.plot(frames, seq/1e9, 'o', color='gray', alpha=0.3, markersize=4,
            label='Raw signal')
    ax.plot(frames, smoothed/1e9, '-', color='lime', linewidth=2,
            label='Smoothed')

    # Highlight cornea frames
    cornea_frames = frames[label == 1]
    if len(cornea_frames):
        ax.plot(cornea_frames, smoothed[cornea_frames]/1e9, 'o',
                color='yellow', markersize=6, zorder=3,
                label=f'Cornea = low plateau (n={label.sum()} frames)')

    ax.axhline(otsu_thresh/1e9, color='cyan', linestyle='--', linewidth=1.5,
               alpha=0.7, label=f'Otsu split ({otsu_thresh/1e9:.3f} GHz) — cornea below')
    if start > 0:
        ax.axvline(start, color='magenta', linestyle='-', linewidth=2, alpha=0.8,
                   label=f'Cornea start (frame {start}) — air before this')

    true_ghz = to_true_shift(gaussian_filter1d(seq.astype(np.float64), 3)[label == 1].mean()/1e9) \
               if label.sum() else float('nan')
    ax.set_title(f'{patient_id} | Point {point_idx} | {diagnosis}  '
                 f'| Cornea: {label.sum()} frames | true shift {true_ghz:.3f} GHz',
                 color='white', fontsize=11, fontweight='bold')
    ax.set_xlabel('Frame (depth)', color='white')
    ax.set_ylabel('Brillouin Shift (GHz)', color='white')
    ax.tick_params(colors='white')
    ax.legend(fontsize=9, facecolor='#0f172a', labelcolor='white')
    ax.grid(True, alpha=0.2)

    plt.tight_layout()
    plt.savefig(save_path, dpi=120, facecolor=fig.get_facecolor(),
                bbox_inches='tight')
    plt.close()


def load_coord_map(path):
    df = pd.read_excel(path)
    df['Patient'] = df['Patient'].astype(str)
    coord_map = {}
    for pid, grp in df.groupby('Patient'):
        coord_map[pid] = grp[['Diagnosis']].reset_index(drop=True)
    return coord_map


def main():
    os.makedirs(PLOT_DIR, exist_ok=True)
    coord_map = load_coord_map(COORD_FILE)
    print(f"Loaded diagnosis map for {len(coord_map)} patients.\n")

    sequences, seg_labels, cls_labels, groups = [], [], [], []
    qc_flags, plateau_means, flagged_points = [], [], []
    label_slopes, steep_points = [], []
    plot_count = 0
    skipped = []

    patient_dirs = sorted(glob.glob(os.path.join(DATA_DIR, '*')))
    patient_dirs = [p for p in patient_dirs if os.path.isdir(p)]

    for pdir in patient_dirs:
        patient_id = os.path.basename(pdir)
        mat_path = os.path.join(pdir, MAT_FILENAME)

        if not os.path.exists(mat_path):
            skipped.append((patient_id, 'no .mat file'))
            continue

        coords = coord_map.get(patient_id)
        if coords is None:
            coords = coord_map.get(patient_id.split()[0])
        if coords is None:
            skipped.append((patient_id, 'no coordinate entry'))
            continue

        diagnosis = coords['Diagnosis'].iloc[0]
        if diagnosis not in LABEL_MAP:
            skipped.append((patient_id, f'unknown diagnosis: {diagnosis}'))
            continue

        mat = scipy.io.loadmat(mat_path)
        if MAT_KEY not in mat:
            skipped.append((patient_id, f'key {MAT_KEY} missing'))
            continue

        arr = mat[MAT_KEY]  # (100, N)
        n_points = min(arr.shape[1], len(coords))
        cls_label = LABEL_MAP[diagnosis]
        n_cornea_frames = []
        n_flagged_this_patient = 0

        for col in range(n_points):
            seq = arr[:, col].astype(np.float32)
            if np.isnan(seq).any():
                continue

            label, boundary, otsu_thresh, start = generate_label(seq)

            # QC: cornea is the LOW region, so a labeled mean that is too
            # HIGH means the label locked onto the aqueous humor instead.
            smoothed_qc = gaussian_filter1d(seq.astype(np.float64), sigma=3)
            plateau_mean = smoothed_qc[label == 1].mean() if label.sum() else 0.0
            # Flatness QC: mean |slope| inside the labeled region. A label that
            # sits on the transition instead of the plateau has a high slope.
            slope_qc = np.abs(np.gradient(smoothed_qc / 1e9))
            label_slope = slope_qc[label == 1].mean() if label.sum() else np.nan
            label_slopes.append(label_slope)
            is_flagged = (plateau_mean / 1e9) > QC_MAX_GHZ
            if is_flagged:
                n_flagged_this_patient += 1
                flagged_points.append((patient_id, col, plateau_mean / 1e9, int(label.sum())))

            sequences.append(seq)
            seg_labels.append(label)
            cls_labels.append(cls_label)
            groups.append(patient_id)
            qc_flags.append(int(is_flagged))
            plateau_means.append(plateau_mean)
            n_cornea_frames.append(label.sum())

            # Save example plots for Ron's verification
            if plot_count < N_EXAMPLE_PLOTS:
                save_path = os.path.join(
                    PLOT_DIR, f'{patient_id}_point{col}_{diagnosis}.png')
                plot_example(seq, label, boundary, otsu_thresh,
                             patient_id, col, diagnosis, save_path, start)
                plot_count += 1

        avg_cornea = np.mean(n_cornea_frames) if n_cornea_frames else 0
        flag_note = f"  [{n_flagged_this_patient} flagged]" if n_flagged_this_patient else ""
        print(f"  {patient_id:15s} {diagnosis:9s}  "
              f"{n_points} points  avg cornea frames: {avg_cornea:.1f}/100{flag_note}")

    if not sequences:
        print("\nNo sequences loaded. Check DATA_DIR.")
        return

    X       = np.stack(sequences)[..., np.newaxis]        # (n, 100, 1)
    Y_seg   = np.stack(seg_labels)                        # (n, 100)
    Y_cls   = np.array(cls_labels, dtype=np.int64)        # (n,)
    groups  = np.array(groups)                            # (n,)

    qc_flags      = np.array(qc_flags, dtype=np.int8)      # (n,) 1 = flagged
    plateau_means = np.array(plateau_means)                # (n,) Hz
    label_slopes  = np.array(label_slopes)                 # (n,) GHz/frame

    # Second QC pass: flatness. Needs the dataset-wide median slope, so it can
    # only run after every sequence has been labeled.
    slope_med = np.nanmedian(label_slopes)
    steep = label_slopes > (QC_SLOPE_MULT * slope_med)
    n_steep_new = int((steep & (qc_flags == 0)).sum())
    qc_flags[steep] = 1

    n_before = len(X)
    n_total_flagged = int(qc_flags.sum())   # capture before filtering
    if EXCLUDE_FLAGGED:
        keep = qc_flags == 0
        X, Y_seg, Y_cls    = X[keep], Y_seg[keep], Y_cls[keep]
        groups             = groups[keep]
        plateau_means      = plateau_means[keep]
        qc_flags           = qc_flags[keep]

    np.savez(OUTPUT_NPZ, X=X, Y_seg=Y_seg, Y_cls=Y_cls, groups=groups,
             qc_flags=qc_flags, plateau_means=plateau_means)

    print("\n" + "=" * 60)
    print(f"Saved: {OUTPUT_NPZ}")
    print(f"X shape:     {X.shape}")
    print(f"Y_seg shape: {Y_seg.shape}  (binary cornea labels per frame)")
    print(f"Y_cls shape: {Y_cls.shape}  (Controls=0 / SKC=1 per sequence)")
    print(f"Patients:    {len(np.unique(groups))}")
    print(f"Avg cornea frames per sequence: "
          f"{Y_seg.mean(axis=1).mean()*100:.1f} frames labeled as cornea")
    if EXCLUDE_FLAGGED:
        print(f"\nExcluded {n_before - len(X)} flagged sequences "
              f"({n_before} -> {len(X)} kept), per Ron's instruction to train "
              f"on the good ones only.")
    pm_ghz = plateau_means / 1e9
    true_ghz = to_true_shift(pm_ghz)
    print(f"Cornea mean (plotted scale): {pm_ghz.mean():.3f} GHz "
          f"(std {pm_ghz.std():.3f})")
    print(f"Cornea mean (TRUE shift, FSR={FSR:.4f}): {true_ghz.mean():.3f} GHz "
          f"(std {true_ghz.std():.3f})   <- Ron: cornea ~5.7 GHz")
    print(f"\nQC value check: {len(flagged_points)}/{n_before} flagged "
          f"(cornea mean > {QC_MAX_GHZ} GHz -> locked onto aqueous)")
    print(f"QC flatness check: {int(steep.sum())}/{n_before} flagged "
          f"(slope > {QC_SLOPE_MULT}x median {slope_med:.5f} -> label on the "
          f"transition, not the plateau); {n_steep_new} of these are new")
    print(f"QC total flagged: {n_total_flagged}/{n_before}")
    if flagged_points:
        print("  Worst offenders:")
        for pid, col, pm, nf in sorted(flagged_points, key=lambda r: -r[2])[:8]:
            print(f"    {pid:15s} point {col:2d}: cornea mean {pm:.3f} GHz, {nf} frames")

    print(f"\nExample plots saved to: {PLOT_DIR}")
    print(f"Send these {N_EXAMPLE_PLOTS} plots to Ron to verify labels look correct")
    print("before training the segmentation model.")
    if skipped:
        print(f"\nSkipped {len(skipped)} patient(s):")
        for pid, reason in skipped:
            print(f"  {pid}: {reason}")
    print("=" * 60)


if __name__ == '__main__':
    main()