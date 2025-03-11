# Use the official Python image as base
FROM python:3.11

# Set the working directory in the container
WORKDIR /app

# Copy the application files
COPY . .

# Install system dependencies required for OpenCV and Tesseract
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    libtesseract-dev \
    liblept5 \
    libgl1-mesa-glx \
    libglib2.0-0 \
    zbar \
    && rm -rf /var/lib/apt/lists/*

# Ensure Tesseract supports Hebrew
RUN mkdir -p /usr/share/tesseract-ocr/4.00/tessdata/ && \
    wget -O /usr/share/tesseract-ocr/4.00/tessdata/heb.traineddata \
    https://github.com/tesseract-ocr/tessdata_best/raw/main/heb.traineddata

# Set environment variable for Tesseract data
ENV TESSDATA_PREFIX="/usr/share/tesseract-ocr/4.00/tessdata"

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Ensure pyzbar is installed
RUN pip install --no-cache-dir pyzbar

# Expose the port the app runs on
EXPOSE 5000

# Command to run the application
CMD ["python", "server.py"]
