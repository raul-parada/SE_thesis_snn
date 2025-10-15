cd ~/SE_thesis_snn

cat > Dockerfile << 'EOF'
# Dockerfile - SNN Log Anomaly Detection
FROM python:3.10-slim

LABEL maintainer="raul.parada@bth.se"
LABEL description="SNN Log Anomaly Detection - Thesis"

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (Docker layer caching)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy all source code
COPY . .

# Create necessary directories
RUN mkdir -p data logs plots

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

# Default command
CMD ["python", "run_pipeline.py"]
EOF

echo "✓ Dockerfile fixed"
