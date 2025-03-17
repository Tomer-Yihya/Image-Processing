# Use a lightweight Python image
FROM python:3.8-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    tesseract-ocr-heb \
    tesseract-ocr-eng \
    libsm6 \
    libxext6 \
    libgl1-mesa-glx \
    libzbar0 \
    libglib2.0-0 \
    libjpeg62-turbo \
    libtiff5-dev \
    libgtk2.0-dev \
    libpng-dev \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
    gcc \
    g++ \
    make \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Ensure Tesseract is accessible
RUN ln -s /usr/bin/tesseract /usr/local/bin/tesseract

# Download required language data manually (Hebrew & English)
RUN mkdir -p /usr/share/tesseract-ocr/4.00/tessdata/ && \
    wget -O /usr/share/tesseract-ocr/4.00/tessdata/heb.traineddata \
    https://github.com/tesseract-ocr/tessdata_best/raw/main/heb.traineddata && \
    wget -O /usr/share/tesseract-ocr/4.00/tessdata/eng.traineddata \
    https://github.com/tesseract-ocr/tessdata_best/raw/main/eng.traineddata

# Set environment variables
ENV TESSDATA_PREFIX="/usr/share/tesseract-ocr/4.00/tessdata/"
ENV TESSERACT_PATH="/usr/bin/tesseract"
ENV PATH="${PATH}:/usr/bin"

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application files
COPY . .

# Expose port
EXPOSE 10000

# Run the server
CMD ["gunicorn", "--bind", "0.0.0.0:10000", "server:app"]
