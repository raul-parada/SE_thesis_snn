cd ~/SE_thesis_snn

cat > Dockerfile << 'EOF'
# Dockerfile - SNN Log Anomaly Detection
FROM python:3.10-slim

LABEL maintainer="raul.parada@bth.se"
LABEL description="SNN Log Anomaly Detection - Thesis"

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    wget \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN mkdir -p data logs plots

ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

CMD ["python", "run_pipeline.py"]
EOF

echo "✓ Dockerfile fixed"
