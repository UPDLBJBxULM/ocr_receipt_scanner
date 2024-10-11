import gspread
from oauth2client.service_account import ServiceAccountCredentials
from datetime import datetime
import os

# Assuming 'credentials.json' is your service account credential file
GOOGLE_SHEET_ID = '1QFVJtpfC7E3WomzZCnmsl4ObeMToNSoY4vuGQ_oPj4s'  # Your Google Sheet ID
UPLOAD_FOLDER_LINK = "https://drive.google.com/drive/folders/1W2Z7Q-3-sefwt9-1Oicj0E6Sd9Lx2Rwx"
PICTURE_FOLDER_LINK = "https://drive.google.com/drive/folders/1gD398GwpRldjfypzzKYTGb-9TqGYDxOe"

# Authenticate and access the sheet
scope = ['https://spreadsheets.google.com/feeds', 'https://www.googleapis.com/auth/drive']
creds = ServiceAccountCredentials.from_json_keyfile_name('C:/Users/ZAINAL/Desktop/Receipt_Scanner/credentials/new_key.json', scope)
client = gspread.authorize(creds)

# Open the specific Google Sheet
sheet = client.open_by_key(GOOGLE_SHEET_ID).worksheet("REKAPREALISASI")

# Function to find the first available blank row
def get_first_blank_row():
    # Get all values in the first column (assuming Tanggal is in column A)
    col_values = sheet.col_values(1)
    
    # Find the first empty row by checking for the first empty cell in column A
    return len(col_values) + 1

# Function to append data to the sheet
def append_to_sheet(total_value, id_rencana, account_list, uraian, judul_laporan):
    # Current timestamp in the format: dd.mm.yyyy hh:mm:ss
    current_time = datetime.now().strftime('%d.%m.%Y %H:%M:%S')

    # Create the row with respective values
    data_to_append = [
        current_time,                 # Column A: Tanggal
        total_value,                  # Column B: Nominal
        id_rencana,                   # Column C: Id Rencana
        UPLOAD_FOLDER_LINK,           # Column D: Scan Nota
        PICTURE_FOLDER_LINK,          # Column E: Gambar Barang
        account_list,                 # Column F: Account List
        '',                           # Column G: Leave blank (no value for this column)
        uraian,                       # Column H: Uraian
        judul_laporan,                # Column I: Judul Laporan
    ]

    # Find the first blank row to avoid overwriting data
    next_row = get_first_blank_row()

    # Append data to the found blank row in the Google Sheet
    sheet.insert_row(data_to_append, next_row)

# Example data
total_value = '50000'  # Example total value
id_rencana = 'SKKO1'   # Example Id Rencana from sheet 'RENCANA'
account_list = '53 / NRT - 6106200200202402 - Pemeliharaan Area Depan Kantor'
uraian = 'Joy'
judul_laporan = 'Judul Dummy'

# Append the example data
append_to_sheet(total_value, id_rencana, account_list, uraian, judul_laporan)