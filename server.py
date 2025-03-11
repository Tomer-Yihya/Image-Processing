from flask import Flask, request, jsonify
from algorithem import extract_text_as_json
import os

app = Flask(__name__)

@app.route('/upload', methods=['POST'])
def upload_image():
    if 'image' not in request.files:
        return jsonify({"error": "No image file provided"}), 400

    image_file = request.files['image']
    if image_file.filename == '':
        return jsonify({"error": "Empty filename"}), 400

    image_path = os.path.join("/tmp", image_file.filename)
    image_file.save(image_path)

    try:
        result = extract_text_as_json(image_path)
        return jsonify(result), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
