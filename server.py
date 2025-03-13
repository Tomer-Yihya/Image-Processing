import os
import json
import logging
import importlib
import time
import requests
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
from flask_cors import CORS
from algorithem import extract_text_as_json

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

@app.route("/status", methods=["GET"])
def status():
    logger.info("✅ /status route accessed!")
    return jsonify({"status": "ok", "message": "Server is running fine"}), 200

@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "Server is running"}), 200

if __name__ == '__main__':
    port = int(os.environ["PORT"])  # Use the actual Render-assigned port
    print(f"🚀 Starting Flask server on port {port}...")
    
    # Debugging: Print all registered routes
    print("📢 Registered routes:")
    for rule in app.url_map.iter_rules():
        print(f"➡ {rule}")
    
    app.run(host="0.0.0.0", port=port, debug=False)
