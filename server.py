from flask import Flask, request, jsonify
import cv2
import numpy as np
import pytesseract
from PIL import Image
import os

app = Flask(__name__)

# ודא שהמשתנה TESSDATA_PREFIX מוגדר כראוי
os.environ["TESSDATA_PREFIX"] = "/usr/share/tesseract-ocr/4.00/tessdata"

def process_image(image):
    try:
        # המרת תמונה לפורמט של OpenCV
        image_np = np.array(image)
        gray = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)  # המרה לגווני אפור

        # שימוש ב-Tesseract עם עברית
        extracted_text = pytesseract.image_to_string(gray, lang="heb")

        # שליחת הטקסט לעיבוד על ידי האלגוריתם
        extracted_data = parse_extracted_text(extracted_text)

        return extracted_data  # החזרת הפלט ללא שינוי
    except pytesseract.TesseractError as e:
        return {"error": f"Tesseract error: {str(e)}"}
    except Exception as e:
        return {"error": f"Unexpected error: {str(e)}"}

def parse_extracted_text(text):
    """ פונקציה המדמה את האלגוריתם הקיים לעיבוד טקסט והחזרת JSON. """
    # כאן יש להכניס את הלוגיקה של האלגוריתם האמיתי
    return {"raw_text": text}  # דוגמה להחזרת טקסט גולמי, יש להחליף באלגוריתם שלך

@app.route("/upload", methods=["POST"])
def upload_image():
    if "image" not in request.files:
        return jsonify({"error": "No image file provided"}), 400

    image_file = request.files["image"]

    # פתיחת התמונה עם PIL
    image = Image.open(image_file)

    # עיבוד התמונה וחילוץ נתונים
    processed_result = process_image(image)

    # הדפסת ה-JSON **כמו שהוא**, ללא שינוי
    print(processed_result)

    return jsonify(processed_result), 200

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
