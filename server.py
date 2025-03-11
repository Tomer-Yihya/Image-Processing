import os
import json
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
from algorithem import extract_text_as_json
import pyzbar

app = Flask(__name__)

# Directory to store uploaded images
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/upload', methods=['POST'])
def upload_image():
    """Handles image uploads and processes them using the extraction algorithm."""
    if 'image' not in request.files:
        return jsonify({"error": "No image file provided"}), 400

    image_file = request.files['image']

    if image_file.filename == '':
        return jsonify({"error": "Empty filename"}), 400

    # Save the image temporarily
    filename = secure_filename(image_file.filename)
    image_path = os.path.join(UPLOAD_FOLDER, filename)
    image_file.save(image_path)

    try:
        # Process the image and extract text as JSON
        extracted_json = extract_text_as_json(image_path)

        # Delete the temporary file after processing
        os.remove(image_path)

        # Return the extracted data
        return extracted_json, 200, {'Content-Type': 'application/json'}

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
