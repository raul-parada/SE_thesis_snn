# Dockerfile for SNN-Based Log Anomaly Detection System
#
# This Dockerfile creates a containerized environment for the Spiking Neural Network
# log anomaly detection pipeline. It includes all necessary dependencies for data
# processing, model training, and evaluation.
#
# Base image: Python 3.10 on Debian slim for minimal size

FROM python:3.10-slim

# Container metadata
LABEL maintainer="raul.parada@bth.se"
LABEL description="SNN Log Anomaly Detection - Thesis Project"
LABEL version="1.0"

# Set working directory for application
WORKDIR /app

# Install system dependencies required for building Python packages
# - build-essential: C/C++ compilers for native extensions
# - git: Version control for potential pip installations from git repositories
# - wget: HTTP utility for downloading resources
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Copy Python dependency specification
COPY requirements.txt .

# Install Python dependencies
# --no-cache-dir reduces image size by not storing pip cache
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source code and configuration
COPY . .

# Create necessary directories for data, logs, and visualization output
RUN mkdir -p data logs plots

# Environment variables
# PYTHONUNBUFFERED: Disable output buffering for real-time logging
# PYTHONPATH: Ensure Python can locate application modules
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

# Default command to execute the main pipeline
CMD ["python", "run_pipeline.py"]
