# Use an official Python runtime as a parent image
FROM python:3.11

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install system dependencies for OpenCV and Tesseract OCR
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    tesseract-ocr-heb \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    libgl1-mesa-glx \
    wget

# Ensure Tesseract tessdata directory exists
RUN mkdir -p /usr/share/tesseract-ocr/4.00/tessdata

# Download Hebrew trained data (if missing)
RUN wget -O /usr/share/tesseract-ocr/4.00/tessdata/heb.traineddata \
    https://github.com/tesseract-ocr/tessdata_best/raw/main/heb.traineddata

# Set Tesseract OCR language data path
ENV TESSDATA_PREFIX="/usr/share/tesseract-ocr/4.00/tessdata"

# Verify installation - check if Hebrew is available
RUN tesseract --list-langs

# Install required Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose port 5000 for Flask
EXPOSE 5000

# Define the command to run the application
CMD ["python", "server.py"]
