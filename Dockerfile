# Use an Ubuntu-based image with Tesseract pre-installed
FROM ubuntu:latest

# Install dependencies
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    libtesseract-dev \
    python3 \
    python3-pip

# Set working directory
WORKDIR /app

# Copy all project files
COPY . /app

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose port 5000
EXPOSE 5000

# Run Flask app
CMD ["python3", "server.py"]
