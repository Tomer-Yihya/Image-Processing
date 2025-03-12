import requests
import json

try:
    import pyzbar
    print("✅ Pyzbar is installed!")
except ModuleNotFoundError:
    print("❌ Pyzbar is NOT installed!")



image_path = "C:/Users/A3DC~1/Desktop/data base/pictures/11.jpg"
url = "https://Just-Do-It-Server.onrender.com/upload"

with open(image_path, "rb") as image_file:
    files = {"image": image_file}
    response = requests.post(url, files=files)

print("Response JSON:")
print(json.dumps(response.json(), ensure_ascii=False, indent=2))
