# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    libtesseract-dev \
    liblept5 \
    wget \
    libgl1-mesa-glx \
    libglib2.0-0

# Set environment variable for Tesseract data
ENV TESSDATA_PREFIX="/usr/share/tesseract-ocr/4.00/tessdata"

# Ensure Hebrew trained data exists
RUN mkdir -p $TESSDATA_PREFIX && \
    wget -O "$TESSDATA_PREFIX/heb.traineddata" \
    https://github.com/tesseract-ocr/tessdata_best/raw/main/heb.traineddata

# Copy the current directory contents into the container at /app
COPY . /app

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install pyzbar

# Make port 5000 available to the world outside this container
EXPOSE 5000

# Run server.py when the container launches
CMD ["python", "server.py"]
