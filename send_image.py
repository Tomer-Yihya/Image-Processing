import requests
import json

def check_module(module_name, display_name):
    try:
        __import__(module_name)
        print(f"✅ {display_name} is installed!")
    except ModuleNotFoundError:
        print(f"❌ {display_name} is NOT installed!")

# Check for required modules
check_module("flask", "Flask")
check_module("flask_cors", "Flask-CORS")
check_module("cv2", "OpenCV")
check_module("numpy", "NumPy")
check_module("matplotlib", "Matplotlib")
check_module("pytesseract", "Pytesseract")
check_module("PIL", "Pillow")
check_module("pyzbar", "Pyzbar")
check_module("werkzeug", "Werkzeug")
check_module("requests", "Requests")

def reverse_hebrew(text):
    return text[::-1] if isinstance(text, str) else text

image_path = "C:/Users/A3DC~1/Desktop/data base/pictures/11.jpg"
url = "https://Just-Do-It-Server.onrender.com/upload"

with open(image_path, "rb") as image_file:
    files = {"image": image_file}
    response = requests.post(url, files=files)

print(f"Response Status Code: {response.status_code}")
print("Raw Response Text:")
print(response.text)

# בדיקה אם התשובה ריקה או לא תקפה כ-JSON
if response.status_code != 200 or not response.text.strip():
    print("❌ Error: No valid JSON response received from the server.")
else:
    try:
        response_json = response.json()
        
        # Reverse all Hebrew fields in the JSON response
        for key in response_json:
            response_json[key] = reverse_hebrew(response_json[key])
        
        print("Response JSON:")
        print(json.dumps(response_json, ensure_ascii=False, indent=2))
    except json.JSONDecodeError as e:
        print(f"❌ JSON Decode Error: {e}")
