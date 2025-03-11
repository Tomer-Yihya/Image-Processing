# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Set the working directory
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . .

# Install system dependencies required for OpenCV, Tesseract, and ZBar
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    libtesseract-dev \
    liblept5 \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libgl1 \
    libzbar0 \
    && rm -rf /var/lib/apt/lists/*


# Ensure Hebrew trained data exists
RUN apt-get update && apt-get install -y wget && \
    mkdir -p /usr/share/tesseract-ocr/4.00/tessdata/ && \
    wget -O /usr/share/tesseract-ocr/4.00/tessdata/heb.traineddata \
    https://github.com/tesseract-ocr/tessdata_best/raw/main/heb.traineddata


# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir pyzbar opencv-python-headless

# Set environment variable for Tesseract data
ENV TESSDATA_PREFIX="/usr/share/tesseract-ocr/4.00/tessdata"

# Expose the port on which the server runs
EXPOSE 5000

# Command to run the application
CMD ["python", "server.py"]
