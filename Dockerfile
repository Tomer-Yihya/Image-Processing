FROM python:3.10

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Set environment variables
ENV FLASK_APP=server.py
ENV FLASK_ENV=production

# Ensure gunicorn runs with the correct settings
CMD ["gunicorn", "-w", "4", "-b", "0.0.0.0:5000", "server:app"]
