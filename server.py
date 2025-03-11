from flask import Flask, request, jsonify
import os
import tempfile
import algorithem

app = Flask(__name__)

@app.route('/upload', methods=['POST'])
def upload_image():
    if 'image' not in request.files:
        return jsonify({"error": "No image file provided"}), 400

    image_file = request.files['image']

    # שמירה בקובץ זמני
    temp_dir = tempfile.gettempdir()
    temp_path = os.path.join(temp_dir, image_file.filename)
    image_file.save(temp_path)

    try:
        # שימוש בפונקציות של algorithem.py לעיבוד התמונה
        processed_fields = algorithem.main_process(temp_path)

        return jsonify(processed_fields), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
