import os
import json
import logging
import importlib
import threading
import time
import requests
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
from flask_cors import CORS
from algorithem import extract_text_as_json

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set Tesseract OCR Path
os.environ["TESSERACT_CMD"] = "/usr/bin/tesseract"
os.environ["TESSDATA_PREFIX"] = "/usr/share/tesseract-ocr/4.00/tessdata/"

app = Flask(__name__)
CORS(app)  # Allows access from any origin

app.config['MAX_CONTENT_LENGTH'] = 5 * 1024 * 1024  # Limit file upload size to 5MB
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route("/status", methods=["GET"])
def status():
    """Returns server health status."""
    logger.info("✅ /status route accessed!")
    return jsonify({"status": "ok", "message": "Server is running fine"}), 200

@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "Server is running"}), 200

@app.route("/upload", methods=['POST'])
def upload_image():
    logger.info("Received a request to /upload")

    if 'image' not in request.files:
        logger.error("❌ No image file provided")
        return jsonify({"error": "No image file provided"}), 400

    image_file = request.files['image']
    if image_file.filename == '':
        logger.error("❌ Empty filename")
        return jsonify({"error": "Empty filename"}), 400

    filename = secure_filename(image_file.filename)
    image_path = os.path.join(UPLOAD_FOLDER, filename)
    image_file.save(image_path)

    try:
        logger.info(f"Processing image: {image_path}")
        extracted_json = extract_text_as_json(image_path)

        # ❗ Delete the image after processing ❗
        os.remove(image_path)
        logger.info(f"✅ Image {filename} processed and deleted")

        if not extracted_json:
            return jsonify({"error": "Failed to process image"}), 500

        return jsonify(json.loads(extracted_json, object_pairs_hook=dict)), 200


    except Exception as e:
        logger.error(f"❌ Error processing image: {str(e)}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 10000))
    print(f"🚀 Starting Flask server on port {port}...")
    print("📢 Registered routes:")
    for rule in app.url_map.iter_rules():
        print(f"➡ {rule}")
    app.run(host="0.0.0.0", port=port, debug=False)
