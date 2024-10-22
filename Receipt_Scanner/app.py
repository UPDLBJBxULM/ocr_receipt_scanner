import os
import cv2
import uuid
import gspread
import logging
import requests
import numpy as np
import imghdr
from ultralytics import YOLO
from dotenv import load_dotenv
from google.cloud import vision
from datetime import datetime, timedelta
from werkzeug.utils import secure_filename
from flask import (
    Flask,
    request,
    jsonify,
    render_template,
    session,
    make_response,
)
from markupsafe import escape
from oauth2client.service_account import ServiceAccountCredentials

# ------------------------ Configuration ------------------------ #

# Load environment variables from .env file
load_dotenv()

# Receipt API configuration
RECEIPT_API_ENDPOINT = os.getenv("RECEIPT_API_ENDPOINT")

# Evidence API configuration
EVIDENCE_API_ENDPOINT = os.getenv("EVIDENCE_API_ENDPOINT")

# YOLO Model Path
YOLO_MODEL_PATH = os.getenv("YOLO_MODEL_PATH")

# Google Credentials Path
GOOGLE_CREDENTIALS_PATH = os.getenv("GOOGLE_CREDENTIALS_PATH")

# Define the scope for the Google Sheets and Google Drive API
SCOPES = [
    "https://spreadsheets.google.com/feeds",
    "https://www.googleapis.com/auth/drive",
]

# Google Sheets configuration
GOOGLE_SHEET_ID = os.getenv("GOOGLE_SHEET_ID")

# Set up Flask app
app = Flask(__name__)
UPLOAD_FOLDER = "./uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = 100 * 1024 * 1024  # Limit file uploads to 100MB

# Set secret key from environment variable
app.secret_key = os.getenv("FLASK_SECRET_KEY", "your_default_secret_key")

# Secure session cookie
app.config.update(
    SESSION_COOKIE_SECURE=True,
    SESSION_COOKIE_HTTPONLY=True,
    SESSION_COOKIE_SAMESITE="Lax",
)

# Ensure the upload directory exists
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ------------------------ Helper Functions ------------------------ #


def init_gspread_client():
    scope = SCOPES
    creds = ServiceAccountCredentials.from_json_keyfile_name(
        GOOGLE_CREDENTIALS_PATH, scope
    )
    client = gspread.authorize(creds)
    return client


def get_rencana_details_from_sheet(rencana_id: str) -> dict:
    client = init_gspread_client()
    sheet = client.open_by_key(GOOGLE_SHEET_ID).worksheet("RENCANA")

    # Get all records
    records = sheet.get_all_records()

    if not records:
        raise Exception("No records found in the RENCANA sheet.")

    for record in records:
        id_value = record.get("id rencana", "")
        if id_value != "":
            try:
                id_int = int(id_value)
                record_id = f"{id_int:05d}"
            except ValueError:
                record_id = str(id_value).strip()
            if record_id == str(rencana_id).strip():
                return {
                    "start_date_ar": record.get("start date A/R", ""),
                    "end_date_ar": record.get("end date A/R", ""),
                    "requestor": record.get("Requestor", ""),
                    "unit": record.get("Unit", ""),
                    "nominal": record.get("Nominal", ""),
                    "id_rencana": record_id,
                }
    raise Exception("ID Rencana not found")


# Function to generate a unique temporary filename
def generate_temp_filename(extension=".jpg") -> str:
    return f"temp_{uuid.uuid4().hex}{extension}"


def set_google_credentials(json_path: str):
    """Set the environment variable for Google Cloud credentials."""
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"Google credentials file not found at {json_path}")
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = json_path


# Initialize the Google Sheets API client
def get_google_sheet():
    client = init_gspread_client()
    return client.open_by_key(GOOGLE_SHEET_ID).worksheet("REKAPREALISASI")


def load_model(model_path: str):
    """Load the YOLOv8 model."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"YOLO model file not found at {model_path}")
    model = YOLO(model_path)
    return model


def get_class_names(model) -> dict:
    """Retrieve class names from the YOLO model."""
    if hasattr(model, "names"):
        return model.names
    elif hasattr(model, "model") and hasattr(model.model, "names"):
        return model.model.names
    else:
        raise AttributeError("Cannot find class names in the YOLO model.")


def visualize_and_extract_total_value(
    image, model, class_names: dict, target_label: str
):
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


def extract_text_from_image(cropped_image) -> str:
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


def append_to_sheet(
    amount: str,
    rencana_id: str,
    account_skkos_id: str,
    uraian: str,
    judulLaporan: str,
    receipt_link: str,
    evidence_links: str,
):
    """Append data to the Google Sheet."""
    scope = SCOPES
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
        receipt_link,  # Column D: Scan Nota
        evidence_links,  # Column E: Gambar Barang
        account_skkos_id,  # Column F: Account List
        "",  # Column G: Leave blank
        uraian,  # Column H: Uraian
        judulLaporan,  # Column I: Judul Laporan
    ]

    # Find the first blank row and append the data
    next_row = len(sheet.col_values(1)) + 1
    sheet.insert_row(data_to_append, next_row)


def query_from_sheet(sheet_name: str, column_idx: int) -> list:
    """Retrieve data from a specific column in the sheet."""
    scope = SCOPES
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


@app.route("/get_rencana_details")
def get_rencana_details():
    rencana_id = escape(request.args.get("rencana_id"))
    try:
        data = get_rencana_details_from_sheet(rencana_id)
        return jsonify(data)
    except Exception as e:
        logger.error(f"Error fetching Rencana details: {str(e)}", exc_info=True)
        return jsonify({"error": str(e)}), 500


@app.route("/submit", methods=["POST"])
def submit_data():
    try:
        # Access form data
        rencana_id = escape(request.form.get("rencana_id"))
        account_skkos_id = escape(request.form.get("account_skkos_id"))
        currency = escape(request.form.get("currency"))
        amount = escape(request.form.get("amount"))
        uraian = escape(request.form.get("uraian"))
        judulLaporan = escape(request.form.get("judulLaporan"))

        # Validate inputs
        if not account_skkos_id:
            return jsonify({"error": "No account_skkos_id provided"}), 400

        # Retrieve necessary data from session
        file_path = session.get("receipt_file_path")
        filename = session.get("receipt_filename")
        content_type = session.get("receipt_content_type")
        extracted_text = session.get("extracted_text")

        if not file_path or not filename or not content_type:
            return jsonify({"error": "No receipt file found to upload."}), 400

        # Prepare additional data to send to Receipt API
        receipt_payload = {
            "accountSKKO": account_skkos_id,
            "extracted_text": extracted_text,
        }

        # Upload the receipt to the external Receipt API
        with open(file_path, "rb") as receipt_file:
            receipt_files = {
                "file": (
                    secure_filename(filename),
                    receipt_file,
                    content_type,
                )
            }
            try:
                receipt_api_response = requests.post(
                    RECEIPT_API_ENDPOINT,
                    data=receipt_payload,
                    files=receipt_files,
                    timeout=500,  # Set timeout for the request
                )
            except requests.Timeout:
                return jsonify({"error": "Receipt API request timed out."}), 504

        if receipt_api_response.status_code == 200:
            receipt_api_data = receipt_api_response.json()
            if receipt_api_data.get("status") == "success":
                receipt_link = receipt_api_data["data"]["fileLink"]
                # Remove the temporary file after upload
                os.remove(file_path)
                # Remove file info from session
                session.pop("receipt_file_path", None)
                session.pop("receipt_filename", None)
                session.pop("receipt_content_type", None)
                session.pop("extracted_text", None)
            else:
                logger.error(f"Receipt API failed: {receipt_api_data.get('message')}")
                return (
                    jsonify({"error": "Failed to upload receipt to Receipt API."}),
                    500,
                )
        else:
            logger.error(
                f"Receipt API responded with status code {receipt_api_response.status_code}"
            )
            return (
                jsonify({"error": "Receipt API responded with an error."}),
                500,
            )

        # Handle evidence files upload
        evidence_links = []
        if "evidence_files" in request.files:
            evidence_files = request.files.getlist("evidence_files")

            # Prepare the payload for the Evidence API
            files_payload = []
            for file in evidence_files:
                if file.filename == "":
                    logger.warning("Skipped a file with no filename.")
                    continue  # Skip files with no name

                # Validate file type using imghdr
                file.seek(0)
                header = file.read(512)
                file.seek(0)
                file_type = imghdr.what(None, header)
                if not file_type:
                    logger.warning(f"Skipped invalid image file: {file.filename}")
                    continue  # Skip invalid image files

                # Validate file size (e.g., max 100MB per file)
                file.seek(0, os.SEEK_END)
                file_length = file.tell()
                if file_length > 100 * 1024 * 1024:  # Maximum Size Evidence
                    logger.warning(
                        f"Skipped file exceeding size limit: {file.filename}"
                    )
                    continue  # Skip files exceeding size limit
                file.seek(0)

                # Append to the files_payload with the correct field name "files"
                files_payload.append(
                    (
                        "files",
                        (
                            secure_filename(file.filename),
                            file.read(),
                            file.content_type,
                        ),
                    )
                )

            if files_payload:
                # Include 'accountSKKO' in the data payload
                data_payload = {
                    "accountSKKO": account_skkos_id,
                }

                try:
                    evidence_api_response = requests.post(
                        EVIDENCE_API_ENDPOINT,
                        data=data_payload,
                        files=files_payload,
                        timeout=500,  # Set timeout for the request
                    )

                    if evidence_api_response.status_code == 200:
                        evidence_api_data = evidence_api_response.json()
                        if evidence_api_data.get("status") == "success":
                            data = evidence_api_data.get("data", {})
                            if isinstance(data, list):
                                for item in data:
                                    file_link = item.get("fileLink")
                                    if file_link:
                                        evidence_links.append(file_link)
                            elif isinstance(data, dict):
                                file_link = data.get("fileLink")
                                if file_link:
                                    evidence_links.append(file_link)
                        else:
                            logger.error(
                                f"Evidence API failed: {evidence_api_data.get('message')}"
                            )
                    else:
                        logger.error(
                            f"Evidence API responded with status code {evidence_api_response.status_code}"
                        )
                except Exception as e:
                    logger.error(
                        f"Exception during upload to Evidence API: {str(e)}",
                        exc_info=True,
                    )

        # Append data to Google Sheet
        append_to_sheet(
            amount,
            rencana_id,
            account_skkos_id,
            uraian,
            judulLaporan,
            receipt_link,
            ", ".join(evidence_links),
        )

        return jsonify(
            success=True,
            message="Data saved successfully! Files uploaded to Google Drive via APIs.",
        )
    except Exception as e:
        logger.error(f"Error in submit_data: {str(e)}", exc_info=True)
        return jsonify(success=False, message=str(e)), 500


# Fetch Id Rencana from Google Sheets
@app.route("/fetch_id_rencana", methods=["GET"])
def fetch_id_rencana():
    """Fetch 'Id Rencana' from the Google Sheet."""
    try:
        client = init_gspread_client()
        sheet = client.open_by_key(GOOGLE_SHEET_ID).worksheet("RENCANA")
        records = sheet.get_all_records()

        id_rencana_data = []
        for record in records:
            id_value = record.get("id rencana", "")
            if id_value != "":
                # Convert to integer if possible
                try:
                    id_int = int(id_value)
                    # Format with leading zeros (e.g., 00001)
                    id_str = f"{id_int:05d}"
                except ValueError:
                    # If conversion fails, use the string as is
                    id_str = str(id_value).strip()
                id_rencana_data.append(id_str)

        return jsonify(id_rencana_data), 200
    except Exception as e:
        logger.error(f"Error fetching Id Rencana: {str(e)}", exc_info=True)
        return jsonify({"error": "Error fetching Id Rencana"}), 500


# Fetch Account SKKO from Google Sheets
@app.route("/fetch_account_skkos", methods=["GET"])
def fetch_account_skkos():
    """Fetch 'Account SKKO' from the Google Sheet."""
    try:
        account_skkos_data = query_from_sheet(
            "ACCOUNTLIST", 1
        )  # Column A of 'ACCOUNTLIST'
        return jsonify(account_skkos_data), 200
    except Exception as e:
        logger.error(f"Error fetching Account SKKOs: {str(e)}", exc_info=True)
        return jsonify({"error": "Error fetching Account SKKOs"}), 500


@app.route("/upload_file", methods=["POST"])
def upload_file():
    """Handle receipt file upload and extract total_value."""
    try:
        # No longer require 'accountSKKO' here

        if "file" not in request.files:
            return jsonify({"error": "No file part"}), 400

        file = request.files["file"]

        # Ensure the file is present and has a filename
        if file.filename == "":
            return jsonify({"error": "No selected file"}), 400

        # Validate file type using imghdr
        file.seek(0)
        header = file.read(512)
        file.seek(0)
        file_type = imghdr.what(None, header)
        if not file_type:
            return jsonify({"error": "Invalid image file."}), 400

        # Validate file size (e.g., max 100MB)
        file.seek(0, os.SEEK_END)
        file_length = file.tell()
        if file_length > 100 * 1024 * 1024:
            return jsonify({"error": "File size exceeds 100MB limit."}), 400
        file.seek(0)

        # Read the image file as a NumPy array for OpenCV
        np_img = np.frombuffer(file.read(), np.uint8)
        image = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

        # Generate a unique temporary filename
        temp_filename = generate_temp_filename()
        file_path = os.path.join(app.config["UPLOAD_FOLDER"], temp_filename)
        cv2.imwrite(file_path, image)

        # Load model and set credentials (load once if not already loaded)
        global model, class_names
        if "model" not in globals():
            model = load_model(YOLO_MODEL_PATH)
            class_names = get_class_names(model)
            set_google_credentials(GOOGLE_CREDENTIALS_PATH)

        # Process the image using YOLO and OCR
        try:
            cropped_img = visualize_and_extract_total_value(
                image, model, class_names, "total_value"
            )

            if cropped_img is not None:
                extracted_text = extract_text_from_image(cropped_img)
                logger.info(f"Extracted Text: {extracted_text}")

                # Store necessary data in session for later use
                session["receipt_file_path"] = file_path
                session["receipt_filename"] = file.filename
                session["receipt_content_type"] = file.content_type
                session["extracted_text"] = extracted_text

                # Return the extracted_text to the frontend
                return (
                    jsonify(
                        {
                            "extracted_text": extracted_text,
                            "message": "Total value extracted successfully.",
                        }
                    ),
                    200,
                )
            else:
                # Remove the temporary file if no total_value detected
                os.remove(file_path)
                return (
                    jsonify(
                        {
                            "error": "No total_value detected in the receipt.",
                        }
                    ),
                    404,
                )
        except Exception as e:
            logger.error(f"Error processing image: {str(e)}", exc_info=True)
            # Remove the temporary file in case of processing error
            if os.path.exists(file_path):
                os.remove(file_path)
            return jsonify({"error": "Error processing image."}), 500

    except Exception as e:
        logger.error(f"Error in upload_file: {str(e)}", exc_info=True)
        return jsonify({"error": "An unexpected error occurred."}), 500


# Remove the '/upload_evidence' route as evidence files will be handled during submission

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5151)
