from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import numpy as np
import os
from algorithem import detect_all_barcodes, process_barcodes_and_extract_fields, process_image


app = Flask(__name__)
CORS(app)  # Allows communication from FlutterFlow

@app.route('/upload', methods=['POST'])
def upload_image():
    if 'image' not in request.files:
        return jsonify({"error": "No image file provided"}), 400

    file = request.files['image']
    np_img = np.frombuffer(file.read(), np.uint8)
    image = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

    # Process image directly without using file path
    detected_barcodes = detect_all_barcodes(image, display_results=False)
    extracted_fields = process_barcodes_and_extract_fields(image, detected_barcodes, display=False)

    return jsonify(extracted_fields)

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
