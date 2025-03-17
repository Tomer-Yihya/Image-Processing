import requests
import json
import importlib

image_path = "C:/Users/A3DC~1/Desktop/data base/pictures/3.jpg"
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
        
        print("\nRaw JSON from server:")
        print(response_json)  # Debug print
        
        # Apply Hebrew text reversal only to the name field
        if "name" in response_json:
            response_json["name"] = reverse_hebrew(response_json["name"])
        
        print("\nResponse JSON:")
        print(json.dumps(response_json, ensure_ascii=False, indent=2))
    except requests.exceptions.HTTPError as http_err:
        print(f"❌ HTTP error occurred: {http_err}")
    except requests.exceptions.RequestException as req_err:
        print(f"❌ Request error occurred: {req_err}")
    except json.JSONDecodeError as json_err:
        print(f"❌ JSON Decode Error: {json_err}")

if __name__ == "__main__":
    send_image()
