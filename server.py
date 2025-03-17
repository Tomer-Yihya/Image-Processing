import os
import json
import logging
import cv2
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
from algorithms import extract_text_as_json  # ייבוא הפונקציה המתאימה

# Ensure Tesseract is correctly set up
TESSERACT_PATH = "/usr/local/bin/tesseract"

# Initialize Flask app
app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("server")

logger.info(f"✅ Tesseract is set to: {TESSERACT_PATH}")

UPLOAD_FOLDER = "uploads"
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Function to check allowed file types
def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route("/status", methods=["GET"])
def status():
    """Health check route"""
    logger.info("✅ /status route accessed!")
    return jsonify({"status": "running"}), 200

@app.route("/upload", methods=["POST"])
def upload_file():
    """Handles image upload and text extraction"""
    logger.info("Received a request to /upload")

    if "image" not in request.files:
        return jsonify({"error": "No image part"}), 400
    
    file = request.files["image"]
    
    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(file_path)
        logger.info(f"Processing image: {file_path}")
        
        try:
            # שימוש באלגוריתם המתקדם לחילוץ מידע מהתמונה
            extracted_json = extract_text_as_json(file_path)

            # מחיקת הקובץ לאחר העיבוד כדי לחסוך מקום
            os.remove(file_path)

            return jsonify(json.loads(extracted_json)), 200
        except Exception as e:
            logger.error(f"❌ Error processing image: {str(e)}")
            return jsonify({"error": f"Failed to process image: {str(e)}"}), 500
    
    return jsonify({"error": "Invalid file format"}), 400

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port, debug=False)
