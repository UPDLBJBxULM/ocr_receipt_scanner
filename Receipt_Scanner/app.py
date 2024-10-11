from flask import Flask, request, jsonify, render_template
import os
import cv2
from ultralytics import YOLO
from google.cloud import vision
import base64
import numpy as np
import gspread
from oauth2client.service_account import ServiceAccountCredentials
from google.oauth2.service_account import Credentials
from werkzeug.utils import secure_filename
from datetime import datetime
import requests
import mimetypes  # Add this import

# ------------------------ Configuration ------------------------ #

# Path to your YOLOv8 model
YOLO_MODEL_PATH = "C:/Users/ZAINAL/Desktop/Receipt_Scanner/models/best.pt"

# Path to the Google Cloud service account credentials
GOOGLE_CREDENTIALS_PATH = (
    "C:/Users/ZAINAL/Desktop/Receipt_Scanner/credentials/new_key.json"
)

# Define the scope for the Google Sheets and Google Drive API
SCOPES = [
    "https://spreadsheets.google.com/feeds",
    "https://www.googleapis.com/auth/drive",
]

# Google Sheets configuration
GOOGLE_SHEET_ID = "1QFVJtpfC7E3WomzZCnmsl4ObeMToNSoY4vuGQ_oPj4s"
UPLOAD_FOLDER_LINK = (
    "https://drive.google.com/drive/folders/1W2Z7Q-3-sefwt9-1Oicj0E6Sd9Lx2Rwx"
)
PICTURE_FOLDER_LINK = (
    "https://drive.google.com/drive/folders/1gD398GwpRldjfypzzKYTGb-9TqGYDxOe"
)

# Set up Flask app
app = Flask(__name__)
UPLOAD_FOLDER = "./uploads"

# Ensure the upload directory exists
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# ------------------------ Helper Functions ------------------------ #


def set_google_credentials(json_path):
    """Set the environment variable for Google Cloud credentials."""
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"Google credentials file not found at {json_path}")
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = json_path


# Initialize the Google Sheets API client
def get_google_sheet():
    scope = [
        "https://spreadsheets.google.com/feeds",
        "https://www.googleapis.com/auth/drive",
    ]
    creds = ServiceAccountCredentials.from_json_keyfile_name(
        GOOGLE_CREDENTIALS_PATH, scope
    )
    client = gspread.authorize(creds)
    return client.open_by_key(GOOGLE_SHEET_ID).worksheet("REKAPREALISASI")


def load_model(model_path):
    """Load the YOLOv8 model."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"YOLO model file not found at {model_path}")
    model = YOLO(model_path)
    return model


def get_class_names(model):
    """Retrieve class names from the YOLO model."""
    if hasattr(model, "names"):
        return model.names
    elif hasattr(model, "model") and hasattr(model.model, "names"):
        return model.model.names
    else:
        raise AttributeError("Cannot find class names in the YOLO model.")


def visualize_and_extract_total_value(image, model, class_names, target_label):
    """Detect the target label, crop the image, and prepare for OCR."""
    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = model(img_rgb)

    highest_conf = 0
    best_box = None

    for result in results:
        boxes = result.boxes.xyxy
        scores = result.boxes.conf
        class_ids = result.boxes.cls

        for box, score, class_id in zip(boxes, scores, class_ids):
            label = class_names.get(int(class_id), None)
            if label == target_label and score > highest_conf:
                highest_conf = score
                best_box = box

    if best_box is not None:
        x1, y1, x2, y2 = map(int, best_box)
        cropped_img = img_rgb[y1:y2, x1:x2]
        return cropped_img
    else:
        return None


def extract_text_from_image(cropped_image):
    """Use Google Cloud Vision to extract text from the cropped image."""
    client = vision.ImageAnnotatorClient()

    _, encoded_image = cv2.imencode(".png", cropped_image)
    content = encoded_image.tobytes()

    image = vision.Image(content=content)
    response = client.text_detection(image=image)

    texts = response.text_annotations

    if response.error.message:
        raise Exception(f"{response.error.message}")

    if texts:
        return texts[0].description
    return ""


def append_to_sheet(amount, rencana_id, account_skkos_id, kegiatan, lpj):
    """Append data to the Google Sheet."""
    scope = [
        "https://spreadsheets.google.com/feeds",
        "https://www.googleapis.com/auth/drive",
    ]
    creds = ServiceAccountCredentials.from_json_keyfile_name(
        GOOGLE_CREDENTIALS_PATH, scope
    )
    client = gspread.authorize(creds)

    # Open the Google Sheet and access the REKAPREALISASI worksheet
    sheet = client.open_by_key(GOOGLE_SHEET_ID).worksheet("REKAPREALISASI")
    current_time = datetime.now().strftime("%d.%m.%Y %H:%M:%S")

    data_to_append = [
        current_time,  # Column A: Tanggal
        amount,  # Column B: Nominal
        rencana_id,  # Column C: Id Rencana
        UPLOAD_FOLDER_LINK,  # Column D: Scan Nota
        PICTURE_FOLDER_LINK,  # Column E: Gambar Barang
        account_skkos_id,  # Column F: Account List
        "",  # Column G: Leave blank
        kegiatan,  # Column H: Uraian
        lpj,  # Column I: Judul Laporan
    ]

    # Find the first blank row and append the data
    next_row = len(sheet.col_values(1)) + 1
    sheet.insert_row(data_to_append, next_row)


def query_from_sheet(sheet_name, column_idx):
    """Retrieve data from a specific column in the sheet."""
    scope = [
        "https://spreadsheets.google.com/feeds",
        "https://www.googleapis.com/auth/drive",
    ]
    creds = ServiceAccountCredentials.from_json_keyfile_name(
        GOOGLE_CREDENTIALS_PATH, scope
    )
    client = gspread.authorize(creds)

    sheet = client.open_by_key(GOOGLE_SHEET_ID).worksheet(sheet_name)
    return sheet.col_values(column_idx)


# ------------------------ Routes ------------------------ #


@app.route("/")
def index():
    """Render the main index page."""
    return render_template("index.html")


@app.route("/upload_file", methods=["POST"])
def upload_file():
    """Handle file uploads and process the receipt."""
    if "file" not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files["file"]

    # Ensure the file is present and has a filename
    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    # Read the image file as a NumPy array for OpenCV
    np_img = np.frombuffer(file.read(), np.uint8)
    image = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

    # Save the file with a temporary name
    temp_filename = "uploaded_temp.jpg"
    file_path = os.path.join(UPLOAD_FOLDER, secure_filename(temp_filename))
    cv2.imwrite(file_path, image)

    # Process the image using YOLO and OCR
    model = load_model(YOLO_MODEL_PATH)
    set_google_credentials(GOOGLE_CREDENTIALS_PATH)
    class_names = get_class_names(model)

    cropped_img = visualize_and_extract_total_value(
        image, model, class_names, "total_value"
    )

    if cropped_img is not None:
        extracted_text = extract_text_from_image(cropped_img)
        print(
            f"Extracted Text: {extracted_text}"
        )  # Log to check what text is being extracted
        return (
            jsonify({"extracted_text": extracted_text, "filename": temp_filename}),
            200,
        )
    else:
        return (
            jsonify({"error": "No total_value detected.", "filename": temp_filename}),
            404,
        )


def upload_to_google_drive(file_path):
    url = "https://fh0kd5s9-3000.asse.devtunnels.ms/api/files/upload"

    # Get the MIME type of the file
    mime_type, _ = mimetypes.guess_type(file_path)
    if mime_type is None:
        mime_type = (
            "application/octet-stream"  # Default to binary if type can't be guessed
        )

    with open(file_path, "rb") as file:
        files = {"file": (os.path.basename(file_path), file, mime_type)}
        response = requests.post(url, files=files)

    if response.status_code == 200:
        data = response.json()
        return data["data"]["fileId"], data["data"]["fileLink"]
    else:
        raise Exception(f"Failed to upload file: {response.text}")


# Modify the submit_data function to include error handling and logging
@app.route("/submit", methods=["POST"])
def submit_data():
    try:
        data = request.json

        rencana_id = data.get("rencana_id")
        account_skkos_id = data.get("account_skkos_id")
        amount = data.get("amount")
        kegiatan = data.get("kegiatan")
        lpj = data.get("lpj")
        filename = data.get("filename")  # Get the temporary file name

        # Validate inputs
        if not account_skkos_id:
            return jsonify({"error": "No account_skkos_id provided"}), 400
        if not filename:
            return jsonify({"error": "No uploaded file found"}), 400

        # Get the current date for the filename
        current_date = datetime.now().strftime("%Y%m%d")  # yyyymmdd

        # Parse account_skkos_id to extract components for renaming the file
        parts = account_skkos_id.split(" - ")
        pos = parts[0][:2]  # First 2 characters
        account = parts[1]  # Number after the first "-"

        # Handle subgl (3 characters after "/")
        if "/" in parts[0]:
            subgl = parts[0].split(" / ")[1][:3]
        else:
            subgl = ""  # Leave blank if not present

        # Build the new filename based on account_skkos_id
        if subgl:
            new_filename = f"{current_date}_{pos}_{account}_{subgl}.jpg"
        else:
            new_filename = f"{current_date}_{pos}_{account}.jpg"

        # Rename the file
        old_file_path = os.path.join(UPLOAD_FOLDER, secure_filename(filename))
        new_file_path = os.path.join(UPLOAD_FOLDER, secure_filename(new_filename))

        if os.path.exists(old_file_path):
            os.rename(old_file_path, new_file_path)  # Rename the file
        else:
            return jsonify({"error": "File does not exist"}), 400

        # Upload the file to Google Drive
        try:
            file_id, file_link = upload_to_google_drive(new_file_path)
        except Exception as upload_error:
            app.logger.error(
                f"Error uploading file to Google Drive: {str(upload_error)}"
            )
            return jsonify({"error": "Failed to upload file to Google Drive"}), 500

        # Append data to Google Sheet
        sheet = get_google_sheet()
        date = datetime.now().strftime("%d.%m.%Y %H:%M:%S")
        barang_link = (
            "https://drive.google.com/drive/folders/1gD398GwpRldjfypzzKYTGb-9TqGYDxOe"
        )

        new_row = [
            date,
            amount,
            rencana_id,
            file_link,  # Use the file_link from Google Drive
            barang_link,
            account_skkos_id,
            "",
            kegiatan,
            lpj,
        ]

        # Find the next empty row
        next_row = len(sheet.col_values(1)) + 1

        # Use batch_update to insert the new row at the correct position
        cell_range = f"A{next_row}:I{next_row}"
        cell_list = sheet.range(cell_range)
        for i, value in enumerate(new_row):
            cell_list[i].value = value

        sheet.update_cells(cell_list)

        return jsonify(
            success=True,
            message=f"Data saved successfully in row {next_row}! File uploaded to Google Drive with ID: {file_id}",
        )
    except Exception as e:
        app.logger.error(f"Error in submit_data: {str(e)}", exc_info=True)
        return jsonify(success=False, message=str(e)), 500


# New route: Fetch Id Rencana from Google Sheets
@app.route("/fetch_id_rencana", methods=["GET"])
def fetch_id_rencana():
    """Fetch 'Id Rencana' from the Google Sheet."""
    id_rencana_data = query_from_sheet("RENCANA", 7)  # Column G of 'RENCANA'
    return jsonify(id_rencana_data), 200


# New route: Fetch Account SKKO from Google Sheets
@app.route("/fetch_account_skkos", methods=["GET"])
def fetch_account_skkos():
    """Fetch 'Account SKKO' from the Google Sheet."""
    account_skkos_data = query_from_sheet("ACCOUNTLIST", 1)  # Column A of 'ACCOUNTLIST'
    return jsonify(account_skkos_data), 200


@app.route("/suggestions", methods=["GET"])
def get_suggestions():
    """Provide suggestions for 'KEGIATAN' and 'LPJ' based on previous entries."""
    column = request.args.get("column")
    if column == "kegiatan":
        data = query_from_sheet("REKAPREALISASI", 8)  # Column H
    elif column == "lpj":
        data = query_from_sheet("REKAPREALISASI", 9)  # Column I
    else:
        return jsonify({"error": "Invalid column"}), 400

    return jsonify({"suggestions": data}), 200


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5151)
