"""
Algorithm 2 - Data Preparation with Spatial Coordinates (plateauXY.py)
=======================================================================

WHAT THIS SCRIPT DOES
----------------------
Same job as dataAutomation.py (Algorithm 1's data script), but this version
extracts THREE pieces of information per scan point instead of one:
  1. Plateau  — the Brillouin frequency shift (corneal stiffness) at this point
  2. X (mm)   — the horizontal position of this scan point on the cornea
  3. Y (mm)   — the vertical position of this scan point on the cornea

WHY THE EXTRA COORDINATES MATTER
---------------------------------
Keratoconus (KC) is a spatially localized disease — the cornea softens most
in the central region (near X=0, Y=0), while the periphery may still look
relatively normal. Algorithm 1 ignores spatial information entirely (just
averages everything). Algorithm 2 keeps the X/Y position of each scan point
so the classifier can learn that central stiffness matters more than
peripheral stiffness for diagnosis.

STRUCTURAL DIFFERENCE FROM dataAutomation.py
---------------------------------------------
Most patient files store Plateau values in the 'Brillouin data' sheet and
the X/Y coordinates in a separate 'XY positions' sheet. This script reads
BOTH sheets per file and joins them by row position (row 0 of Plateau goes
with row 0 of XY, etc.). One patient (20230124) is an exception where both
Plateau and XY are stored in the same sheet (Sheet1).

INPUT  : Per-patient .xlsx files in FOLDER_PATH, each containing:
           - 'Brillouin data' sheet with a 'Plateau' column
           - 'XY positions' sheet with 'X (mm)' and 'Y (mm)' columns
           (exception: 20230124 has both in Sheet1)
         SKC_names.xlsx — diagnosis key (Controls / SKC patient lists)

OUTPUT : combined_data.xlsx
           Patient | Plateau | X (mm) | Y (mm) | Diagnosis
           (one row per spatial scan point, all 30 patients combined)
"""

import pandas as pd
import glob
import os

# ── CONFIGURATION ─────────────────────────────────────────────────────────
# The name of the stiffness column to extract from each patient's Excel file.
COLUMN_NAME = 'Plateau'

# Path pattern matching all per-patient Excel files in the data folder.
# glob.glob() will expand the * wildcard and return every .xlsx found.
FOLDER_PATH = '/Users/minhnguyen/try-scipy/Minh Data/Patient Data/*.xlsx'

# Path to the diagnosis key file containing two columns:
#   Controls: patient identifiers for healthy control eyes
#   SKC:      patient identifiers for subclinical keratoconus eyes
SKC_FILE = '/Users/minhnguyen/try-scipy/Minh Data/SKC_names.xlsx'

# Two patient files deviate from the standard Excel structure (wrong sheet
# name, different column names). We map their filenames to their correct
# (sheet_name, column_name) so they can still be read correctly.
# All other files use: sheet='Brillouin data', column='Plateau'.
OUTLIERS = {
    'Metrics at selected points 20230124.xlsx': ('Sheet1', 'Brillouin shifts of plateau before'),
    'Metrics at selected points 20230420.xlsx': ('Brillouin data', 'Plateau before'),
}
# ──────────────────────────────────────────────────────────────────────────


# ── LOAD DIAGNOSIS KEY ─────────────────────────────────────────────────────
# Read the diagnosis spreadsheet and split it into two Python lists.
skc_df = pd.read_excel(SKC_FILE)

# dropna() removes empty cells at the bottom of each column.
# str() handles IDs that may be stored as integers in Excel.
# strip() removes any accidental leading/trailing whitespace.
controls_names = [str(n).strip() for n in skc_df['Controls'].dropna()]
skc_names      = [str(n).strip() for n in skc_df['SKC'].dropna()]


# ── DIAGNOSIS LOOKUP FUNCTION ──────────────────────────────────────────────
def get_diagnosis(filename):
    """
    Determine whether a patient file belongs to Controls or SKC by matching
    the filename against the two diagnosis lists.

    The matching is word-level rather than exact-string to handle cases like:
      filename  : 'Metrics at selected points 20220628 Left.xlsx'
      identifier: '20220628 Left'
      SKC entry : '20220628 Left'   <- correct match
      SKC entry : '20220628'        <- should NOT match (different patient)
      SKC entry : '20220628 Right'  <- should NOT match (different eye)

    Step by step:
    1. Strip the fixed prefix and extension to get the patient ID string.
    2. Split that string into individual words for word-level comparison.
    3. For each name in the SKC list, check if the date portion appears
       anywhere in the identifier (fast pre-filter).
    4. If the date matches, confirm every word in the identifier also
       appears in that name entry (prevents partial matches).
    5. If found in SKC, return 'SKC'. Then try Controls with same logic.
    6. If no match in either list, return 'Unknown'.

    Args:
        filename: bare filename (not full path), e.g.
                  'Metrics at selected points 20211014.xlsx'

    Returns:
        'SKC', 'Controls', or 'Unknown'
    """
    # Strip prefix and extension to isolate just the patient identifier.
    # e.g. 'Metrics at selected points 20220628 Left.xlsx' -> '20220628 Left'
    identifier = filename.replace('Metrics at selected points ', '').replace('.xlsx', '').strip()

    # Split into individual words for word-level matching.
    # e.g. '20220628 Left' -> ['20220628', 'Left']
    identifier_words = identifier.split()

    # Search the SKC list
    for name in skc_names:
        # Use only the first word (the date) as a fast pre-filter to avoid
        # running the expensive all() check on every name in the list.
        name_date = name.split(' ')[0]
        if name_date in identifier:
            # Confirm every word in our identifier appears in this name entry.
            # This prevents '20220628' from incorrectly matching '20220628 Left'.
            if all(word in name for word in identifier_words):
                return 'SKC'

    # Search the Controls list using the same logic
    for name in controls_names:
        name_date = name.split(' ')[0]
        if name_date in identifier:
            if all(word in name for word in identifier_words):
                return 'Controls'

    # No match found in either list
    return 'Unknown'


# ── MAIN DATA EXTRACTION LOOP ──────────────────────────────────────────────
# Find all .xlsx files in the patient data folder.
all_files = glob.glob(FOLDER_PATH)

# Collect one small DataFrame per patient; concatenate all of them at the end.
all_data = []

for file_path in all_files:
    # Extract just the filename from the full directory path.
    filename = os.path.basename(file_path)

    # Excel creates temporary lock files starting with '~$' when a workbook
    # is currently open. These cannot be read by pandas — skip them.
    if filename.startswith('~$'):
        print(f"Skipping temp file: {filename}")
        continue

    # ── PLATEAU DATA ─────────────────────────────────────────────────────
    # Determine the correct sheet and column for this file's plateau data.
    # Known outliers have non-standard formatting; all others use defaults.
    if filename in OUTLIERS:
        sheet_name, col_name = OUTLIERS[filename]
    else:
        sheet_name, col_name = 'Brillouin data', COLUMN_NAME

    # Read the plateau (stiffness) column from the appropriate sheet.
    df_plateau = pd.read_excel(file_path, sheet_name=sheet_name)
    df_plateau.columns = df_plateau.columns.str.strip()   # remove trailing spaces
    plateau_column = df_plateau[col_name].copy()

    # ── X/Y COORDINATES ──────────────────────────────────────────────────
    # Most files keep X/Y in a separate 'XY positions' sheet.
    # Exception: 20230124 stores X/Y in the same Sheet1 as the plateau data,
    # so we reuse the DataFrame we already loaded rather than reading again.
    if filename == 'Metrics at selected points 20230124.xlsx':
        df_xy = df_plateau   # X and Y are in the same sheet as Plateau
    else:
        # Read the X/Y coordinates from the separate sheet.
        # Row N of this sheet corresponds to row N of the plateau column,
        # so they can be joined directly by index position.
        df_xy = pd.read_excel(file_path, sheet_name='XY positions')
        df_xy.columns = df_xy.columns.str.strip()

    # ── ASSEMBLE PER-PATIENT CHUNK ────────────────────────────────────────
    # Strip prefix and extension to get just the patient identifier string.
    date_id = filename.replace('Metrics at selected points ', '').replace('.xlsx', '')

    # Look up diagnosis for this patient
    diagnosis = get_diagnosis(filename)

    # Build one row per spatial scan point with all four pieces of data.
    # .values extracts the underlying numpy array so all arrays are the
    # same length and no pandas index alignment issues arise.
    chunk = pd.DataFrame({
        'Patient':   date_id,
        'Plateau':   plateau_column.values,   # stiffness at this scan point (GHz)
        'X (mm)':    df_xy['X (mm)'].values,  # horizontal cornea position in mm
        'Y (mm)':    df_xy['Y (mm)'].values,  # vertical cornea position in mm
        'Diagnosis': diagnosis                 # 'Controls', 'SKC', or 'Unknown'
    })

    all_data.append(chunk)
    print(f"{filename} — {diagnosis}")


# ── COMBINE AND SAVE ───────────────────────────────────────────────────────
# Stack all per-patient DataFrames into one combined table.
# ignore_index=True resets the row index to run continuously (0, 1, 2, ...)
# instead of repeating the index from each patient's individual chunk.
combined = pd.concat(all_data, ignore_index=True)

# Drop any rows with a missing Plateau value (e.g. blank rows in the Excel
# file). We do not drop on X/Y NaN here because missing coordinates would
# also fail the Plateau check in most cases; the training script handles
# any remaining NaNs in all three columns before fitting.
combined = combined.dropna(subset=['Plateau'])

# Save to disk. algorithimTrainingXY.py reads this file as its input.
combined.to_excel('combined_data.xlsx', index=False)
print(f"\nDone! {len(combined)} total rows saved to combined_data.xlsx")