import requests
import json
import importlib
from collections import OrderedDict

#image_path = "C:/Users/A3DC~1/Desktop/data base/pictures/7.jpg"
#image_path = "C:/Users/A3DC~1/Desktop/data base/pictures/6.jpg"
#image_path = "C:/Users/A3DC~1/Desktop/data base/pictures/5.jpg"
#image_path = "C:/Users/A3DC~1/Desktop/data base/pictures/4.jpg"
#image_path = "C:/Users/A3DC~1/Desktop/data base/pictures/3.jpg"
#image_path = "C:/Users/A3DC~1/Desktop/data base/pictures/2.jpg"
image_path = "C:/Users/A3DC~1/Desktop/data base/pictures/1.jpg"
server_url = "https://Just-Shoot-It-Server.onrender.com"

REQUIRED_MODULES = {
    "flask": "Flask",
    "flask_cors": "Flask-CORS",
    "cv2": "OpenCV",
    "numpy": "NumPy",
    "matplotlib": "Matplotlib",
    "pytesseract": "Pytesseract",
    "PIL": "Pillow",
    "pyzbar": "Pyzbar",
    "werkzeug": "Werkzeug",
    "requests": "Requests"
}

def check_module(module_name, display_name):
    """Checks if a module is installed on the client machine."""
    try:
        importlib.import_module(module_name)
        print(f"✅ {display_name} is installed!")
    except ModuleNotFoundError:
        print(f"❌ {display_name} is NOT installed!")

def check_all_modules():
    """Check all required modules."""
    print("\n🔍 Checking local (client-side) module installation:")
    for module, name in REQUIRED_MODULES.items():
        check_module(module, name)

def reverse_hebrew(text):
    """Reverse Hebrew text for proper display."""
    return text[::-1] if isinstance(text, str) else text

def send_image():
    """Send an image to the server and process the response."""
    url = f"{server_url}/upload"
    
    try:
        with open(image_path, "rb") as image_file:
            files = {"image": image_file}
            response = requests.post(url, files=files, timeout=120)

        response.raise_for_status()
        response_json = response.json()

        # Reverse Hebrew text fields
        for key in response_json:
            if isinstance(response_json[key], str):
                response_json[key] = reverse_hebrew(response_json[key])

        # Ensure JSON fields are printed in the correct order
        ordered_response = OrderedDict([
            ("date", response_json.get("date", "")),
            ("name", response_json.get("name", "")),
            ("person_id", response_json.get("person_id", "")),
            ("case_id", response_json.get("case_id", ""))
        ])

        print("\nResponse JSON:")
        print(json.dumps(ordered_response, ensure_ascii=False, indent=2))

    except requests.exceptions.HTTPError as http_err:
        print(f"❌ HTTP error occurred: {http_err}")
    except requests.exceptions.RequestException as req_err:
        print(f"❌ Request error occurred: {req_err}")
    except json.JSONDecodeError as json_err:
        print(f"❌ JSON Decode Error: {json_err}")

if __name__ == "__main__":
    #check_all_modules()
    send_image()
