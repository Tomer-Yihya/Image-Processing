# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    libtesseract-dev \
    liblept5 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libglib2.0-dev \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Ensure Hebrew trained data exists
RUN mkdir -p /usr/share/tesseract-ocr/4.00/tessdata/ && \
    wget -O /usr/share/tesseract-ocr/4.00/tessdata/heb.traineddata \
    https://github.com/tesseract-ocr/tessdata_best/raw/main/heb.traineddata

# Set environment variable for Tesseract data
ENV TESSDATA_PREFIX="/usr/share/tesseract-ocr/4.00/tessdata/"

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Ensure pyzbar is installed
RUN pip install pyzbar

# Expose port 5000 for the Flask app
EXPOSE 5000

# Define the command to run the server
CMD ["python", "server.py"]
