# Use Ubuntu as base image
FROM ubuntu:20.04

# Set environment variables to avoid interactive prompts
ENV DEBIAN_FRONTEND=noninteractive
ENV TESSDATA_PREFIX=/usr/share/tesseract-ocr/4.00/tessdata

# Update and install required packages
RUN apt-get update && apt-get install -y \
    python3 python3-pip python3-dev \
    tesseract-ocr libtesseract-dev libleptonica-dev \
    libgl1-mesa-glx libglib2.0-0 wget \
    && rm -rf /var/lib/apt/lists/*

# Download Hebrew Tesseract language model
RUN wget -O /usr/share/tesseract-ocr/4.00/tessdata/heb.traineddata \
    https://github.com/tesseract-ocr/tessdata_best/raw/main/heb.traineddata

# Set up working directory
WORKDIR /app

# Copy project files
COPY . .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose port
EXPOSE 10000

# Run the server
CMD ["gunicorn", "-w", "1", "-b", "0.0.0.0:10000", "server:app"]
