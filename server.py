from flask import Flask, request, jsonify
import os
import tempfile
import algorithem  # Importing the custom algorithm module

app = Flask(__name__)

@app.route('/upload', methods=['POST'])
def upload_image():
    """
    Handles image upload, processes it using the algorithm, extracts relevant fields, 
    and returns the extracted data in JSON format.
    """
    if 'image' not in request.files:
        return jsonify({"error": "No image file provided"}), 400

    image_file = request.files['image']

    # Save the uploaded image to a temporary file
    temp_dir = tempfile.gettempdir()
    temp_path = os.path.join(temp_dir, image_file.filename)
    image_file.save(temp_path)

    try:
        # **1. Image Processing** - Cropping and preparing the sticker
        resized = algorithem.process_image(temp_path)

        # **2. Barcode Detection** - Identifying all barcodes in the sticker
        detected_barcodes = algorithem.detect_all_barcodes(resized, display_results=False)

        # **3. Field Extraction** - Extracting relevant fields based on barcode locations
        extracted_fields = algorithem.process_barcodes_and_extract_fields(resized, detected_barcodes, display=False)

        # **4. Adjust Hebrew Name Orientation (Right-to-Left)**
        if 'name' in extracted_fields:
            extracted_fields['name'] = extracted_fields['name'][::-1]

        return jsonify(extracted_fields), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    # Run the Flask server, allowing access from any network interface
    app.run(host='0.0.0.0', port=5000)
