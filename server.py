import os
import cv2
import pytesseract
import numpy as np
from flask import Flask, request, jsonify

app = Flask(__name__)

# Debug: Print TESSDATA_PREFIX value to verify it's set correctly
print("TESSDATA_PREFIX:", os.getenv("TESSDATA_PREFIX"))

@app.route('/upload', methods=['POST'])
def upload_image():
    if 'image' not in request.files:
        return jsonify({"error": "No image part in request"}), 400
    
    file = request.files['image']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    
    image_np = np.frombuffer(file.read(), np.uint8)
    image = cv2.imdecode(image_np, cv2.IMREAD_COLOR)
    if image is None:
        return jsonify({"error": "Invalid image format"}), 400
    
    # Convert image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    try:
        extracted_text = pytesseract.image_to_string(gray)
    except pytesseract.TesseractError as e:
        return jsonify({"error": f"Tesseract error: {str(e)}"}), 500
    
    return jsonify({"result": extracted_text})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
