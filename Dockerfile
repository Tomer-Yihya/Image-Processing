# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . .

# Install system dependencies required for OpenCV, Tesseract, and barcode scanning
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    libtesseract-dev \
    liblept5 \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libgl1 \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Ensure Hebrew trained data exists
RUN mkdir -p /usr/share/tesseract-ocr/4.00/tessdata/ && \
    wget -O /usr/share/tesseract-ocr/4.00/tessdata/heb.traineddata \
    https://github.com/tesseract-ocr/tessdata_best/raw/main/heb.traineddata

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose the port the app runs on
EXPOSE 5000

# Run the application with Gunicorn
CMD ["gunicorn", "-b", "0.0.0.0:5000", "server:app"]
