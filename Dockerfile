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

# Set Tesseract OCR language data path
ENV TESSDATA_PREFIX="/usr/share/tesseract-ocr/4.00/tessdata"

# Ensure Hebrew trained data exists
RUN if [ ! -f "$TESSDATA_PREFIX/heb.traineddata" ]; then \
        wget -O "$TESSDATA_PREFIX/heb.traineddata" \
        https://github.com/tesseract-ocr/tessdata_best/raw/main/heb.traineddata; \
    fi

# Verify Tesseract installation
RUN tesseract --list-langs || echo "Tesseract installation check failed"

# Install required Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose port 5000 for Flask
EXPOSE 5000

# Define the command to run the application
CMD ["python", "server.py"]
