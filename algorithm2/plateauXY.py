import pandas as pd
import glob
import os

# ------------------CONFIGURATION CONSTANTS------------------
COLUMN_NAME = 'Plateau'
FOLDER_PATH = '/Users/minhnguyen/try-scipy/Minh Data/Patient Data/*.xlsx'
SKC_FILE = '/Users/minhnguyen/try-scipy/Minh Data/SKC_names.xlsx'

OUTLIERS = {
    'Metrics at selected points 20230124.xlsx': ('Sheet1', 'Brillouin shifts of plateau before'),
    'Metrics at selected points 20230420.xlsx': ('Brillouin data', 'Plateau before'),
}

# ------------------LOAD DIAGNOSIS KEY------------------
skc_df = pd.read_excel(SKC_FILE)
controls_names = [str(n).strip() for n in skc_df['Controls'].dropna()]
skc_names = [str(n).strip() for n in skc_df['SKC'].dropna()]

# ------------------DIAGNOSIS LOOKUP FUNCTION------------------
def get_diagnosis(filename):
    identifier = filename.replace('Metrics at selected points ', '').replace('.xlsx', '').strip()
    identifier_words = identifier.split()

    for name in skc_names:
        name_date = name.split(' ')[0]
        if name_date in identifier:
            if all(word in name for word in identifier_words):
                return 'SKC'

    for name in controls_names:
        name_date = name.split(' ')[0]
        if name_date in identifier:
            if all(word in name for word in identifier_words):
                return 'Controls'

    return 'Unknown'

# ------------------EXTRACT DATA------------------
all_files = glob.glob(FOLDER_PATH)
all_data = []

for file_path in all_files:
    filename = os.path.basename(file_path)

    # Skip temporary lock files
    if filename.startswith('~$'):
        print(f"⏭ Skipping temp file: {filename}")
        continue

    # Get correct sheet and column name for Plateau
    if filename in OUTLIERS:
        sheet_name, col_name = OUTLIERS[filename]
    else:
        sheet_name, col_name = 'Brillouin data', COLUMN_NAME

    # Read Plateau data
    df_plateau = pd.read_excel(file_path, sheet_name=sheet_name)
    df_plateau.columns = df_plateau.columns.str.strip()
    plateau_column = df_plateau[col_name].copy()

    # Read X (mm) and Y (mm)
    # For 20230124, X and Y are in the same Sheet1 as Plateau so we reuse df_plateau
    # For all other files, X and Y are in the separate XY positions sheet
    if filename == 'Metrics at selected points 20230124.xlsx':
        df_xy = df_plateau
    else:
        df_xy = pd.read_excel(file_path, sheet_name='XY positions')
        df_xy.columns = df_xy.columns.str.strip()

    date_id = filename.replace('Metrics at selected points ', '').replace('.xlsx', '')
    diagnosis = get_diagnosis(filename)

    chunk = pd.DataFrame({
        'Patient':   date_id,
        'Plateau':   plateau_column.values,
        'X (mm)':    df_xy['X (mm)'].values,
        'Y (mm)':    df_xy['Y (mm)'].values,
        'Diagnosis': diagnosis
    })
    all_data.append(chunk)
    print(f"✅ {filename} — {diagnosis}")

# ------------------COMBINE AND SAVE------------------
combined = pd.concat(all_data, ignore_index=True)
combined = combined.dropna(subset=['Plateau'])
combined.to_excel('combined_data.xlsx', index=False)
print(f"\nDone! {len(combined)} total rows saved to combined_data.xlsx")