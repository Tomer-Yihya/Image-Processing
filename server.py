import os
import json
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
from algorithem import extract_text_as_json
import pyzbar
from werkzeug.middleware.proxy_fix import ProxyFix

app = Flask(__name__)
app.wsgi_app = ProxyFix(app.wsgi_app)
app.config['MAX_CONTENT_LENGTH'] = 5 * 1024 * 1024  # מאפשר העלאת קבצים עד 5MB

# Directory to store uploaded images
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.before_request
def before_request():
    request.environ['wsgi.input_terminated'] = False

@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "Server is running"}), 200

@app.route('/upload', methods=['POST'])
def upload_image():
    print("Received a request to /upload")
    if 'image' not in request.files:
        print("❌ No image file provided")
        return jsonify({"error": "No image file provided"}), 400

    image_file = request.files['image']
    if image_file.filename == '':
        print("❌ Empty filename")
        return jsonify({"error": "Empty filename"}), 400

    filename = secure_filename(image_file.filename)
    image_path = os.path.join(UPLOAD_FOLDER, filename)
    image_file.save(image_path)
    print(f"✅ Image saved at {image_path}")
    print("Uploaded files:", os.listdir(UPLOAD_FOLDER))

    try:
        print(f"Processing image: {image_path}")
        extracted_json = extract_text_as_json(image_path)
        print(f"✅ Extraction complete: {extracted_json}")
        os.remove(image_path)

        if not extracted_json:
            return jsonify({"error": "Failed to process image"}), 500
        
        return jsonify(json.loads(extracted_json)), 200

    except Exception as e:
        print(f"❌ Error processing image: {str(e)}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
