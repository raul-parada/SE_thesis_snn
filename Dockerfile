# Dockerfile
FROM python:3.10-slim

LABEL maintainer="your.email@university.se"
LABEL description="SNN Log Anomaly Detection - Thesis Reproducibility Container"
LABEL version="1.0"

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (for Docker layer caching)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY . .

# Install package in editable mode
RUN pip install -e .

# Create necessary directories
RUN mkdir -p data logs plots

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV CUDA_VISIBLE_DEVICES=""

# Default command (can be overridden)
CMD ["python", "src/run_pipeline.py"]

# Expose port for Jupyter (optional)
EXPOSE 8888
