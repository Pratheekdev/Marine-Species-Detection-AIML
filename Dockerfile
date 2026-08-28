FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY . .

# Environment variables
ENV PYTHONUNBUFFERED=1

# Render injects $PORT at runtime — default to 10000 for local Docker runs
ENV PORT=10000
EXPOSE 10000

CMD gunicorn wsgi:app --bind 0.0.0.0:$PORT --workers 1 --threads 4 --timeout 300
