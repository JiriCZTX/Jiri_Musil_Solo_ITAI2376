# Dockerfile for Workforce Intelligence System
# Jiri Musil | ITAI 2376 | Spring 2026

FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ && \
    rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Default: run full training + demo
CMD ["python", "main.py", "--mode", "full"]
