"""
Algorithm 1 - Data Automation (dataAutomation.py)
==================================================

WHAT THIS SCRIPT DOES
----------------------
Walks a folder of per-patient Excel files, extracts the Brillouin plateau
column from each one, looks up whether that patient is a Control or has
Subclinical Keratoconus (SKC), and stacks everything into one master
spreadsheet (combined_plateau.xlsx) for use by algorithimTraining.py.

This is the data pipeline step. It handles the messy real-world details of
inconsistently formatted Excel files so the ML training script can assume
clean, consistent input.

INPUT  : A folder of per-patient .xlsx files named like:
           "Metrics at selected points 20211014.xlsx"
         A diagnosis key file (SKC_names.xlsx) with two columns:
           Controls | SKC
           listing which patient identifiers belong to which group.

OUTPUT : combined_plateau.xlsx
           Patient | Plateau | Diagnosis
           (one row per spatial measurement point, all 30 patients combined)
"""

import pandas as pd
import glob
import os

# ── CONFIGURATION ─────────────────────────────────────────────────────────
# The name of the column to extract from each patient's Excel file.
# This is the Lorentzian-fitted Brillouin plateau shift value (in GHz),
# which represents the stiffness of the cornea at that spatial scan point.
COLUMN_NAME = 'Plateau'

# Path pattern that matches all per-patient Excel files in the data folder.
# glob.glob() will expand the wildcard (*) and return every .xlsx file found.
FOLDER_PATH = '/Users/minhnguyen/try-scipy/Minh Data/Patient Data/*.xlsx'

# Path to the diagnosis key file. This Excel sheet has two columns:
#   Controls: lists the patient identifiers for healthy control eyes
#   SKC: lists the patient identifiers for subclinical keratoconus eyes
SKC_FILE = '/Users/minhnguyen/try-scipy/Minh Data/SKC_names.xlsx'

# Some patient Excel files do not follow the standard format used by the rest.
# For those, we record the correct sheet name and column name here.
# Structure: { filename: (sheet_name, column_name) }
# If a file is not in this dictionary, the standard format is used:
#   sheet_name = 'Brillouin data', column_name = COLUMN_NAME ('Plateau')
OUTLIERS = {
    'Metrics at selected points 20230124.xlsx': ('Sheet1', 'Brillouin shifts of plateau before'),
    'Metrics at selected points 20230420.xlsx': ('Brillouin data', 'Plateau before'),
}
# ──────────────────────────────────────────────────────────────────────────


# ── LOAD DIAGNOSIS KEY ─────────────────────────────────────────────────────
# Read the SKC_names.xlsx file and build two separate Python lists:
# one for Controls and one for SKC patients.
skc_df = pd.read_excel(SKC_FILE)

# dropna() removes any empty rows at the bottom of the column.
# str() converts patient ID numbers to strings (some may be stored as integers).
# strip() removes any accidental leading/trailing whitespace in the cell values.
controls_names = [str(n).strip() for n in skc_df['Controls'].dropna()]
skc_names      = [str(n).strip() for n in skc_df['SKC'].dropna()]


# ── DIAGNOSIS LOOKUP FUNCTION ──────────────────────────────────────────────
def get_diagnosis(filename):
    """
    Look up whether a patient's file belongs to the Controls or SKC group.

    The matching logic handles cases where the filename and the name in
    SKC_names.xlsx don't match exactly word-for-word. For example, a file
    named '20220628 Left.xlsx' should match the SKC entry '20220628 Left'
    but NOT '20220628' or '20220628 Right'.

    Step by step:
    1. Strip the common prefix 'Metrics at selected points ' and the .xlsx
       extension to get just the patient identifier (e.g. '20220628 Left').
    2. Split the identifier into individual words so we can check each word
       separately (e.g. ['20220628', 'Left']).
    3. For each name in the SKC list, check if the date portion of that name
       appears anywhere in our identifier — this is a fast pre-filter.
    4. If step 3 passes, check that every word in our identifier appears in
       that name entry — this prevents partial matches across similar IDs.
    5. If a full match is found in SKC, return 'SKC'. Otherwise try Controls.
    6. If no match is found in either list, return 'Unknown'.

    Args:
        filename: the bare filename (not full path), e.g.
                  'Metrics at selected points 20211014.xlsx'

    Returns:
        'SKC', 'Controls', or 'Unknown'
    """
    # Remove the fixed prefix and extension to extract just the patient ID
    identifier = filename.replace('Metrics at selected points ', '').replace('.xlsx', '').strip()

    # Split the identifier into individual words for word-level matching.
    # e.g. '20220628 Left' -> ['20220628', 'Left']
    identifier_words = identifier.split()

    # Check SKC list first
    for name in skc_names:
        # The first word of the name entry is always the date (e.g. '20220628').
        # Use it as a fast pre-filter before doing the more expensive word check.
        name_date = name.split(' ')[0]

        # If the date portion appears anywhere in our identifier, this name
        # entry is a candidate for a match.
        if name_date in identifier:
            # Confirm every word in our identifier also appears in the name entry.
            # This prevents '20220628' from matching '20220628 Left' incorrectly.
            if all(word in name for word in identifier_words):
                return 'SKC'

    # Same logic for the Controls list
    for name in controls_names:
        name_date = name.split(' ')[0]
        if name_date in identifier:
            if all(word in name for word in identifier_words):
                return 'Controls'

    # No match found in either list — flag for manual review
    return 'Unknown'


# ── MAIN DATA EXTRACTION LOOP ──────────────────────────────────────────────
# Find every .xlsx file in the patient data folder.
# glob.glob() expands the wildcard (*) and returns the full paths.
all_files = glob.glob(FOLDER_PATH)

# This list will collect one small DataFrame per patient.
# After the loop, we concatenate them all into one big table.
all_data = []

for file_path in all_files:
    # os.path.basename() extracts just the filename from the full path.
    # e.g. '/Users/.../Metrics at selected points 20211014.xlsx'
    #       -> 'Metrics at selected points 20211014.xlsx'
    filename = os.path.basename(file_path)

    # Excel creates temporary lock files starting with '~$' when a workbook
    # is open. Skip these — they cannot be read by pandas.
    if filename.startswith('~$'):
        print(f"Skipping temp file: {filename}")
        continue

    # Determine which sheet and column to read.
    # If this file is a known outlier, use its custom format.
    # Otherwise use the standard format shared by all other files.
    if filename in OUTLIERS:
        sheet_name, col_name = OUTLIERS[filename]
    else:
        sheet_name, col_name = 'Brillouin data', COLUMN_NAME

    # Read the specified sheet from this patient's Excel file.
    # sheet_name tells pandas which tab inside the workbook to open.
    df = pd.read_excel(file_path, sheet_name=sheet_name)

    # Strip whitespace from column names in case any have trailing spaces.
    # Without this, 'Plateau ' (with trailing space) would not match 'Plateau'.
    df.columns = df.columns.str.strip()

    # Extract just the plateau column — one value per spatial scan point.
    target_column = df[col_name].copy()

    # Strip the standard prefix and extension to get the patient ID string,
    # e.g. 'Metrics at selected points 20211014.xlsx' -> '20211014'
    date_id = filename.replace('Metrics at selected points ', '').replace('.xlsx', '')

    # Look up whether this patient is Controls or SKC
    diagnosis = get_diagnosis(filename)

    # Build a small 3-column DataFrame for this patient.
    # Each row is one spatial scan point (one Brillouin plateau measurement).
    chunk = pd.DataFrame({
        'Patient':   date_id,            # patient identifier string
        'Plateau':   target_column.values,  # array of plateau readings
        'Diagnosis': diagnosis           # 'Controls', 'SKC', or 'Unknown'
    })

    all_data.append(chunk)
    print(f"{filename} — {diagnosis}")


# ── COMBINE AND SAVE ───────────────────────────────────────────────────────
# Stack all patient DataFrames into one big table.
# ignore_index=True resets the row index so it runs 0, 1, 2, ... continuously
# instead of repeating 0, 1, 2 from each individual patient's chunk.
combined = pd.concat(all_data, ignore_index=True)

# Remove any rows where the Plateau value was missing (NaN).
# This can happen if a patient file had blank rows at the bottom of its column.
combined = combined.dropna(subset=['Plateau'])

# Save to disk. algorithimTraining.py reads this file as its input.
combined.to_excel('combined_plateau.xlsx', index=False)
print(f"\nDone! {len(combined)} total rows saved to combined_plateau.xlsx")