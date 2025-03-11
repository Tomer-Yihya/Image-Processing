import requests

# Define the server URL
SERVER_URL = "https://Tomer-and-Daniela-Image-Processing-server.onrender.com/upload"

# Path to the image to be sent
image_path = "C:/Users/A3DC~1/Desktop/data base/pictures/11.jpg"

# Open the image file and send it to the server
with open(image_path, "rb") as image_file:
    files = {"image": image_file}
    response = requests.post(SERVER_URL, files=files)

# Print the response status code
print(f"Status Code: {response.status_code}\n")

# Print the exact JSON response
try:
    response_json = response.json()
    print("Response JSON:")
    print(response.text)  # Printing as raw text to show exact server output
except requests.exceptions.JSONDecodeError:
    print("Error: Response is not valid JSON")
