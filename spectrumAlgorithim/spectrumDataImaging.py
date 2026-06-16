"""
2D SPATIAL OVERVIEW — Brillouin Peak Detector

This is the bird's-eye view half of your pipeline. While the depth profiler
looks frame-by-frame down the Z-axis to measure stiffness over time, this
script ignores depth completely and focuses on verifying the camera captured
the Brillouin peaks correctly.

Think of it as looking straight down through a translucent book:
you can see all the ink at once, superimposed into one bright 2D picture.
Then it hunts for the two tallest spikes of light in that combined image.

The output is a visual dashboard: a heat map with crosshairs marking the peaks,
and a spectral profile graph showing the peak separation.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from nptdms import TdmsFile

# Default reconstruction settings for the top-view orientation.
DEFAULT_ORIENTATION = 'row'   # options: 'row', 'col'
DEFAULT_TRANSPOSE = True
# Note: DEFAULT_RESHAPE was removed because the script now figures out 
# the dimensions dynamically based on the file size.


def load_tdms(file_path):
    """Load the TDMS file and return all channels from the data group.
    
    This unpacks the raw EMCCD camera data, the first step in the funnel.
    It includes a safety check in case the recording software changes the 
    default group name from 'Image' to something else.
    """
    tdms_file = TdmsFile.read(file_path)
    
    # Safely look for the 'Image' group, or fallback to the first available group
    if 'Image' in tdms_file:
        return tdms_file['Image'].channels()
    else:
        # Fallback: grab the first group in the file if 'Image' isn't found
        first_group = tdms_file.groups()[0]
        print(f"Warning: 'Image' group not found. Using '{first_group.name}' instead.")
        return first_group.channels()


def build_cube(channels, orientation, transpose_block):
    """Reconstruct the 3D cube from raw EMCCD channel data dynamically.

    Each channel contains one flattened 2D block of spatial pixels. The block
    is dynamically reshaped to the actual number of recorded frames, optionally
    transposed, and inserted into the 3D cube for downstream analysis.
    This box contains all depth slices and their spatial information.
    """
    nrows = len(channels)
    
    # 1. Figure out the total data points in one channel
    raw_len = len(channels[0][:])
    
    # 2. Assume ncolumns = nrows for now (standard square scan)
    ncolumns = nrows 
    
    # 3. DYNAMICALLY calculate how many frames are in this specific file.
    # This prevents the script from crashing if a scan is longer or shorter than 100 frames.
    nframes = raw_len // ncolumns
    
    if raw_len % ncolumns != 0:
        print(f"Warning: Data length ({raw_len}) does not divide perfectly by columns ({ncolumns}). Some data may be truncated to prevent a crash.")

    # Create the empty 3D box based on the dynamic frame count
    cube = np.zeros((nrows, ncolumns, nframes), dtype=np.float32)

    for n, ch in enumerate(channels):
        raw = ch[:].astype(np.float32)
        
        # Trim any excess broken data at the end of the file that doesn't fit into a perfect frame
        valid_length = nframes * ncolumns
        raw = raw[:valid_length]
        
        # Reshape the data block based on the dynamic frame count
        if transpose_block:
            block = raw.reshape((nframes, ncolumns)).T
        else:
            block = raw.reshape((ncolumns, nframes))

        if orientation == 'row':
            cube[n, :, :] = block
        else:
            cube[:, n, :] = block

    return cube, channels


def find_brillouin_peaks(spectral_profile):
    """Find the two main Brillouin peaks in the summed spectral profile.

    This is the peak hunter: it uses a progressive search strategy to locate
    the two tallest spikes of light, even when the signal is weak.
    
    Strategy:
    1. Look for massive, obvious spikes (high prominence).
    2. If the data is weak, relax the rules and search again.
    3. If still no luck, accept any peaks with minimum spacing.
    4. As a last resort, just grab the two largest values.
    
    Returns the pixel column indices of the left and right Brillouin peaks.
    """
    peaks, props = find_peaks(spectral_profile, distance=3, prominence=1e5)
    if len(peaks) < 2:
        peaks, props = find_peaks(spectral_profile, distance=3, prominence=5e4)
    if len(peaks) < 2:
        peaks, props = find_peaks(spectral_profile, distance=3)

    if len(peaks) >= 2:
        if len(props.get('prominences', [])) >= 2:
            top2_idx = np.argsort(props['prominences'])[::-1][:2]
            top2_cols = np.sort(peaks[top2_idx])
        else:
            top2_idx = np.argsort(spectral_profile[peaks])[::-1][:2]
            top2_cols = np.sort(peaks[top2_idx])
        return int(top2_cols[0]), int(top2_cols[1]), peaks, props

    top2_idx = np.argsort(spectral_profile)[::-1][:2]
    top2_cols = np.sort(top2_idx)
    return int(top2_cols[0]), int(top2_cols[1]), peaks, props


def summarize_orientation(cube, orientation):
    # Squash all frames into a single bright 2D picture.
    # This is like looking straight down through a translucent book: you see all the ink at once.
    image = cube.sum(axis=2)
    
    # Create the 1D line graph: total light intensity along the spectral axis.
    if orientation == 'row':
        spectral_profile = image.sum(axis=0)
    else:
        spectral_profile = image.sum(axis=1)

    peak_L, peak_R, peaks, props = find_brillouin_peaks(spectral_profile)
    return image, spectral_profile, peak_L, peak_R


def run_viewer(file_path):
    # Run the full 2D spatial overview pipeline and display the results.
    # This is the manager function that creates the visual dashboard.
    channels = load_tdms(file_path)
    
    # Notice we no longer pass a hardcoded reshape dimension here
    cube, channels_data = build_cube(channels, DEFAULT_ORIENTATION, DEFAULT_TRANSPOSE)
    image, spectral_profile, peak_L, peak_R = summarize_orientation(cube, DEFAULT_ORIENTATION)

    # Find the spatial row position of each peak (where the light is brightest).
    row_L = int(np.argmax(image[:, peak_L]))
    row_R = int(np.argmax(image[:, peak_R]))
    sep = peak_R - peak_L  # The pixel distance between the two Brillouin peaks

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(f'Brillouin EMCCD — 2D View', fontsize=12, fontweight='bold', color='white')

    # LEFT PLOT: Heat map of the squashed 2D image with crosshairs at the peak locations.
    # This is the bird's-eye view: you can see the two brightest spots where the Brillouin light concentrates.
    ax = axes[0]
    im = ax.imshow(image, aspect='auto', cmap='inferno', origin='upper')
    ax.axvline(peak_L, color='cyan', linestyle='--', linewidth=1.5,
               label=f'Left peak  (col {peak_L}, row {row_L})')
    ax.axvline(peak_R, color='lime', linestyle='--', linewidth=1.5,
               label=f'Right peak (col {peak_R}, row {row_R})')
    ax.plot(peak_L, row_L, 'c^', markersize=10, zorder=5)  # Cyan marker at left peak
    ax.plot(peak_R, row_R, 'g^', markersize=10, zorder=5)  # Lime marker at right peak
    ax.set_title('EMCCD Image (all frames summed)', fontsize=11, color='white')
    ax.set_xlabel('Pixel column  →  spectral axis')
    ax.set_ylabel('Pixel row  →  spatial axis')
    ax.legend(loc='upper right', fontsize=9)
    plt.colorbar(im, ax=ax, label='Intensity (counts)')

    # RIGHT PLOT: The 1D line graph showing the two Brillouin peaks and their separation.
    # This is the spectral fingerprint: the exact pixel distance between the peaks tells you
    # the material stiffness at this particular location in the cornea.
    ax = axes[1]
    ax.plot(spectral_profile, '-o', markersize=3, color='steelblue', label='Spectral profile')
    ax.axvline(peak_L, color='cyan', linestyle='--', linewidth=1.5,
               label=f'Left peak   col = {peak_L}')
    ax.axvline(peak_R, color='lime', linestyle='--', linewidth=1.5,
               label=f'Right peak  col = {peak_R}')
    ax.plot(peak_L, spectral_profile[peak_L], 'c^', markersize=12, zorder=5)
    ax.plot(peak_R, spectral_profile[peak_R], 'g^', markersize=12, zorder=5)
    ax.annotate(f'sep = {sep} px', xy=((peak_L + peak_R) / 2, spectral_profile[peak_R]),
                ha='center', va='bottom', fontsize=10, color='white',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='gray', alpha=0.7))
    ax.set_title('Spectral Profile (summed over all spatial rows)', fontsize=11, color='white')
    ax.set_xlabel('Pixel column')
    ax.set_ylabel('Total intensity (counts)')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_facecolor('#1a1a2e')

    fig.patch.set_facecolor('#16213e')
    for a in axes:
        a.tick_params(colors='white')
        a.xaxis.label.set_color('white')
        a.yaxis.label.set_color('white')

    plt.tight_layout()
    # CRITICAL: block=False tells the computer, "Draw this picture on screen, but don't pause."
    # This magic line allows your master controller to keep running and immediately generate
    # the 1D Depth Profile graph without waiting for you to close this window.
    plt.show(block=False) 
    
    return {
        # Dynamically returns the actual shape instead of relying on hardcoded numbers
        "cube_shape": (image.shape[0], image.shape[1], len(channels_data[0][:]) // len(channels_data)),
        "peak_L": peak_L,
        "peak_R": peak_R,
        "sep": sep
    }