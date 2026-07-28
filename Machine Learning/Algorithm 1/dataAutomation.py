import pandas as pd
import glob
import os

#------------------CONFIGURATION CONSTANTS------------------
#primary independent measurment column that we are isolating
COLUMN_NAME = 'Plateau'
#global filepaths for locating the patient data and the diagnosis key for each patient
FOLDER_PATH = '/Users/minhnguyen/try-scipy/Minh Data/Patient Data/*.xlsx'
SKC_FILE = '/Users/minhnguyen/try-scipy/Minh Data/SKC_names.xlsx'
#formatting Exceptions: Sheets/columns formatting of these to excel sheets are different than the rest of the excel sheets
OUTLIERS = {
    'Metrics at selected points 20230124.xlsx': ('Sheet1', 'Brillouin shifts of plateau before'),
    'Metrics at selected points 20230420.xlsx': ('Brillouin data', 'Plateau before'),
}

##------------------LOADING DIAGNOSIS KEY------------------
#reads the SKC file and makes two list
skc_df = pd.read_excel(SKC_FILE)
#takes all of the control patients and puts it into one list, converting patient number to list then deleting any spaces
controls_names = [str(n).strip() for n in skc_df['Controls'].dropna()]
#takes all of SKC patients and puts it into another list
skc_names = [str(n).strip() for n in skc_df['SKC'].dropna()]

#------------------DIAGNOSIS LOOKUP FUNCTION------------------
def get_diagnosis(filename):
    #strip the common prefix and file extension to get the patient identifier
    identifier = filename.replace('Metrics at selected points ', '').replace('.xlsx', '').strip()
    #breaks a string into a list of indivudal words so we can check each word individually against the SKC names instead of mmatching the whole string at once
    identifier_words = identifier.split()

    #initializes a loop that goes through the new skc_names list 
    for name in skc_names:
        #grabs the first word of the file name
        name_date = name.split(' ')[0]
        #if the patient number appears anywhere in an entry of SKC name we continue
        if name_date in identifier:
            #if every single word passes  then it returns True and returns SKC
            if all(word in name for word in identifier_words):
                return 'SKC'
            
    #if the SKC loop finds nothing the Controls loop runs the exact same logic
    for name in controls_names:
        name_date = name.split(' ')[0]
        if name_date in identifier:
            if all(word in name for word in identifier_words):
                return 'Controls'
            
    #if nothing is found Unknown is returned
    return 'Unknown'

#searches the file path and returns a list of every excel sheet it finds in the folder
all_files = glob.glob(FOLDER_PATH)
#an empty list that is created to collect data as the loop runs, the patient's data is appended to this list to later stack them into one excel sheet
all_data = []

#starts the loop running through the list of ever excel sheet we found in the folder
for file_path in all_files:
    #strips the folder path and just gives us the specific file name
    filename = os.path.basename(file_path)

    #checks if a file is currently open, then a skip message is printed and we continue
    if filename.startswith('~$'):
        print(f"⏭ Skipping temp file: {filename}")
        continue

    #check if the file name is one of the two known outliers, if so grab the special sheet and column from the OUTLIERS dictionary. If no, use standard format
    if filename in OUTLIERS:
        sheet_name, col_name = OUTLIERS[filename]
    else:
        sheet_name, col_name = 'Brillouin data', COLUMN_NAME

    #reads the excel file and tells pandas which tab inside the excel sheet to read
    df = pd.read_excel(file_path, sheet_name=sheet_name)
    df.columns = df.columns.str.strip()

    
    target_column = df[col_name].copy()
    date_id = filename.replace('Metrics at selected points ', '').replace('.xlsx', '')
    diagnosis = get_diagnosis(filename)

    chunk = pd.DataFrame({
        'Patient': date_id,
        'Plateau': target_column.values,
        'Diagnosis': diagnosis
    })
    all_data.append(chunk)
    print(f"✅ {filename} — {diagnosis}")

combined = pd.concat(all_data, ignore_index=True)
combined = combined.dropna(subset=['Plateau'])
combined.to_excel('combined_plateau.xlsx', index=False)
print(f"\nDone! {len(combined)} total rows saved to combined_plateau.xlsx")