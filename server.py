from flask import Flask, request, jsonify
import cv2
import numpy as np
import os

app = Flask(__name__)

@app.route('/upload', methods=['POST'])
def upload_image():
    if 'image' not in request.files:
        return jsonify({"error": "No image file found in request"}), 400
    
    file = request.files['image']
    np_img = np.frombuffer(file.read(), np.uint8)
    image = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

    if image is None:
        return jsonify({"error": "Failed to decode image"}), 400
    
    processed_result = process_image(image)
    return jsonify({"result": processed_result})

def process_image(image):
    # כאן נכנס האלגוריתם שלך לעיבוד התמונה
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return "Image processed successfully"

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
