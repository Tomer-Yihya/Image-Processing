# Use an official image that includes Tesseract OCR
FROM ubuntu:20.04

# Set environment variables to avoid interactive prompts
ENV DEBIAN_FRONTEND=noninteractive

# Update and install required packages
RUN apt-get update && apt-get install -y \
    python3 python3-pip python3-dev \
    tesseract-ocr libtesseract-dev libleptonica-dev \
    libgl1-mesa-glx libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Set up working directory
WORKDIR /app

# Copy project files
COPY . .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose port (Render will assign the actual one)
EXPOSE 10000

# Run the server
CMD ["gunicorn", "-w", "1", "-b", "0.0.0.0:$PORT", "server:app"]

