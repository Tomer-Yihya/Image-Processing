# Use an official Python runtime as a parent image
FROM python:3.11

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install system dependencies and Tesseract with Hebrew support
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    tesseract-ocr-heb \
    libtesseract-dev \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

# Set the Tesseract-OCR data path
ENV TESSDATA_PREFIX="/usr/share/tesseract-ocr/4.00/tessdata"

# Verify that Tesseract is installed correctly
RUN tesseract --version

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose port 5000 for Flask app
EXPOSE 5000

# Define environment variable
ENV FLASK_APP=server.py

# Run the application
CMD ["python", "server.py"]
