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

# Set Tesseract path
os.environ["TESSERACT_CMD"] = "/usr/bin/tesseract"

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)  # Allows access from any origin

app.config['MAX_CONTENT_LENGTH'] = 5 * 1024 * 1024  # Limit file upload size to 5MB
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Required modules for checking
REQUIRED_MODULES = {
    "flask": "Flask",
    "flask_cors": "Flask-CORS",
    "cv2": "OpenCV",
    "numpy": "NumPy",
    "matplotlib": "Matplotlib",
    "pytesseract": "Pytesseract",
    "PIL": "Pillow",
    "pyzbar": "Pyzbar",
    "werkzeug": "Werkzeug",
    "requests": "Requests"
}

SERVER_URL = "https://Just-Shoot-It-Server.onrender.com"
PING_INTERVAL = 300  # 5 minutes

def keep_server_alive():
    """Sends a periodic request to prevent server from sleeping."""
    def ping():
        while True:
            try:
                response = requests.get(f"{SERVER_URL}/status", timeout=5)
                logger.info(f"🔄 Ping to keep server alive: {response.status_code}")
            except requests.RequestException as e:
                logger.error(f"⚠️ Ping failed: {e}")
            time.sleep(PING_INTERVAL)

    thread = threading.Thread(target=ping, daemon=True)
    thread.start()

# Start the keep-alive thread
keep_server_alive()

@app.route("/status", methods=["GET"])
def status():
    """Returns server health status."""
    logger.info("✅ /status route accessed!")
    return jsonify({"status": "ok", "message": "Server is running fine"}), 200

@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "Server is running"}), 200

@app.route("/config", methods=["GET"])
def config():
    """Returns server configuration details."""
    return jsonify({"port": os.environ.get("PORT", "10000"), "max_upload_size": "5MB"}), 200

@app.route("/check_modules", methods=["GET"])
def check_modules():
    """Returns a list of installed and missing modules."""
    missing_modules = []
    installed_modules = {}
    for module, name in REQUIRED_MODULES.items():
        try:
            importlib.import_module(module)
            installed_modules[name] = "Installed ✅"
        except ModuleNotFoundError:
            installed_modules[name] = "Not Installed ❌"
            missing_modules.append(name)

    return jsonify({"installed_modules": installed_modules, "missing_modules": missing_modules}), 200

@app.route('/upload', methods=['POST'])
def upload_image():
    """Handles image upload and processing."""
    logger.info("📥 Received a request to /upload")

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
        logger.info(f"🔍 Processing image: {image_path}")
        extracted_json = extract_text_as_json(image_path)

        # ❗ Delete the image after processing ❗
        os.remove(image_path)
        logger.info(f"✅ Image {filename} processed and deleted")

        if not extracted_json:
            return jsonify({"error": "Failed to process image"}), 500

        return jsonify(json.loads(extracted_json)), 200

    except Exception as e:
        logger.error(f"❌ Error processing image: {str(e)}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 10000))
    print(f"🚀 Starting Flask server on port {port}...")

    print("📢 Registered routes:")
    for rule in app.url_map.iter_rules():
        print(f"➡ {rule}")  # Prints all available routes

    app.run(host="0.0.0.0", port=port, debug=False)
