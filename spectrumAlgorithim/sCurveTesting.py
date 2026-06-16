"""
Brillouin S-curve analyzer for cornea/aqueous boundary detection.

Think of this script as a multi-stage funnel:

1. Unpack raw TDMS camera data into a 3D cube of depth frames.
2. Collapse each frame to a 1D profile and measure the distance
   between the two Brillouin peaks.
3. Smooth that 100-point shift curve, find the steepest drop,
   and isolate the cornea plateau before the liquid begins.
4. Apply a coarse ±2σ filter to reject large camera noise.
5. Apply IQR filtering to catch smaller anomalies inside the plateau.
6. Compute the final clean mean using only the surviving, verified
   cornea points, and visualize the result.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d
from nptdms import TdmsFile

DEFAULT_ORIENTATION = 'row'
DEFAULT_TRANSPOSE   = True
DEFAULT_RESHAPE     = (100, 50)

# ── Step 1: Load and reconstruct cube ─────────────────────────────────────
# This function turns the raw TDMS trace data into a 3D volume:
#   depth frames x horizontal pixels x vertical pixels
# It is the funnel's first stage: unpack the raw sensor stream into a
# physically shaped cube of 100 slices for downstream frame-by-frame analysis.
def load_cube(file_path, orientation, reshape_dims, transpose_block):
    tdms = TdmsFile.read(file_path)
    channels = list(tdms['Image'].channels())
    n_ch     = len(channels)
    n_frames = len(channels[0][:]) // n_ch
    cube     = np.zeros((n_ch, n_ch, n_frames), dtype=np.float32)

    for n, ch in enumerate(channels):
        raw   = ch[:].astype(np.float32)
        block = raw.reshape(reshape_dims)
        if transpose_block:
            block = block.T
        if orientation == 'row':
            cube[n, :, :] = block
        else:
            cube[:, n, :] = block

    return cube, n_frames

# ── Step 2: Extract Brillouin shift per frame ─────────────────────────────
# This stage collapses each frame into a 1D spectral line, finds the two
# strongest Brillouin peaks, and measures the distance between them.
# The result is a 1x100 stiffness curve: high values are tissue, low values are liquid.
def extract_shift_per_frame(cube, orientation):
    n_frames = cube.shape[2]
    shifts   = np.full(n_frames, np.nan)

    for f in range(n_frames):
        frame_2d = cube[:, :, f]
        profile  = frame_2d.sum(axis=0) if orientation == 'row' else frame_2d.sum(axis=1)

        # Attempt a strong-peak detection first, then relax thresholds if needed.
        peaks, props = find_peaks(profile, distance=3, prominence=1e3)
        if len(peaks) < 2:
            peaks, props = find_peaks(profile, distance=3, prominence=1e2)
        if len(peaks) < 2:
            peaks, _ = find_peaks(profile, distance=3)

        if len(peaks) >= 2:
            top2 = peaks[np.argsort(profile[peaks])[::-1][:2]]
            top2 = np.sort(top2)
            shifts[f] = top2[1] - top2[0]

    return shifts

# ── Step 3: Find the Cliff & Isolate the Plateau ──────────────────────────
# This stage identifies the physical boundary between tissue and liquid.
# It smooths the 100-point shift curve, finds the steepest downward drop,
# and treats everything before that drop as the cornea plateau candidate region.
def isolate_cornea_plateau(shifts):
    valid = ~np.isnan(shifts)
    frames = np.arange(len(shifts))
    
    valid_frames = frames[valid]
    s_valid = shifts[valid]

    # Smooth the curve to find the macro S-shape
    smoothed = gaussian_filter1d(s_valid, sigma=3)
    
    # Find the steepest negative slope (the drop into the aqueous humor)
    slope = np.gradient(smoothed)
    interface_idx = np.argmin(slope)
    
    interface_frame = valid_frames[interface_idx]
    
    # ML FEATURE: Extract the exact steepness of the drop
    steepness = slope[interface_idx]

    plateau_end_idx = max(0, interface_idx - 3)
    candidate_frames = valid_frames[:plateau_end_idx]
    candidate_vals   = s_valid[:plateau_end_idx]

    plateau_mean  = np.mean(candidate_vals)
    plateau_std   = np.std(candidate_vals) # ML FEATURE: Tissue Uniformity

    lower_bound = plateau_mean - 2 * plateau_std
    upper_bound = plateau_mean + 2 * plateau_std

    return lower_bound, upper_bound, plateau_mean, plateau_std, interface_frame, candidate_frames, candidate_vals, steepness

# ── Step 4: IQR outlier detection ─────────────────────────────────────────
# This is the fine-tooth comb: detect lingering anomalies inside the plateau
# that survived the initial ±2σ camera-noise filter.
def detect_outliers_iqr(values, multiplier=1.5):
    q1  = np.percentile(values, 25)
    q3  = np.percentile(values, 75)
    iqr = q3 - q1
    return (values < q1 - multiplier * iqr) | (values > q3 + multiplier * iqr)

# ── Step 5: Final assembly and visualization ─────────────────────────────
def run_analyzer(file_path):
    cube, n_frames = load_cube(file_path, DEFAULT_ORIENTATION, DEFAULT_RESHAPE, DEFAULT_TRANSPOSE)
    frames = np.arange(n_frames)
    shifts = extract_shift_per_frame(cube, DEFAULT_ORIENTATION)
    
    lower, upper, p_mean, p_std, edge_frame, c_frames, c_vals, steepness = isolate_cornea_plateau(shifts)
    
    candidate_mask = np.isin(frames, c_frames)
    other_frames = frames[~candidate_mask]
    other_vals   = shifts[~candidate_mask]

    in_band = (c_vals >= lower) & (c_vals <= upper)
    out_band = ~in_band

    band_frames = c_frames[in_band]
    band_values = c_vals[in_band]
    out_bound_frames = c_frames[out_band]
    out_bound_values = c_vals[out_band]

    iqr_mask = detect_outliers_iqr(band_values)
    clean_frames = band_frames[~iqr_mask]
    clean_values = band_values[~iqr_mask]
    iqr_outlier_frames = band_frames[iqr_mask]
    iqr_outlier_values = band_values[iqr_mask]

    clean_mean = np.mean(clean_values) if len(clean_values) > 0 else 0

    fig, ax = plt.subplots(figsize=(13, 6))

    # 1. Ignored data (Aqueous liquid and the drop itself)
    ax.scatter(other_frames, other_vals, color='gray', alpha=0.3, s=30, label='Post-boundary / Liquid')

    # 2. Clean plateau points — the final surviving cornea data
    ax.scatter(clean_frames, clean_values, color='lime', s=45, zorder=3,
               label=f'Plateau inliers  (n={len(clean_values)})')

    # 3. IQR outliers — inside the plateau band but statistically anomalous
    ax.scatter(iqr_outlier_frames, iqr_outlier_values, color='orange',
               s=70, marker='D', zorder=4,
               label=f'IQR outliers in band  (n={len(iqr_outlier_values)})')

    # 4. Outside-bound outliers — before the cliff but outside the ±2σ tissue band
    ax.scatter(out_bound_frames, out_bound_values, color='red',
               s=70, marker='x', linewidths=2, zorder=4,
               label=f'Outside-bound outliers  (n={len(out_bound_frames)})')

    # Visual Aids
    ax.axvline(edge_frame, color='magenta', linestyle=':', linewidth=2, label=f'Detected Edge (Frame {edge_frame})')
    ax.axhline(upper, color='cyan', linestyle='--', linewidth=1.5, alpha=0.7)
    ax.axhline(lower, color='cyan', linestyle='--', linewidth=1.5, alpha=0.7)
    ax.axhline(clean_mean, color='white', linestyle='-', linewidth=2, label=f'Clean plateau mean = {clean_mean:.4f} px')

    ax.set_facecolor('#1a1a2e')
    fig.patch.set_facecolor('#16213e')
    ax.tick_params(colors='white')
    ax.xaxis.label.set_color('white')
    ax.yaxis.label.set_color('white')
    ax.title.set_color('white')
    ax.grid(True, alpha=0.2)
    ax.legend(loc='upper right', facecolor='#0f172a', edgecolor='#334155', labelcolor='white', fontsize=8.5)
    ax.set_title(f'Strict Cornea Plateau — {file_path.split("/")[-1]}', fontsize=12, fontweight='bold')
    ax.set_xlabel('Frame (depth)')
    ax.set_ylabel('Brillouin Shift (peak separation, pixels)')

    plt.tight_layout()
    plt.show(block=True) 

    # We now return ALL the ML features here!
    return {
        "edge_frame": edge_frame,
        "raw_mean": p_mean,
        "clean_mean": clean_mean,
        "outliers_removed": len(iqr_outlier_values) + len(out_bound_values),
        "plateau_std": p_std,
        "steepness": steepness
    }