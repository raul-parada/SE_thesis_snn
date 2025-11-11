# Makefile for SNN-Based Log Anomaly Detection System
#
# This Makefile provides convenient command shortcuts for building, testing,
# and running the anomaly detection pipeline. It supports both local execution
# and Docker-based workflows.
#
# Usage: make <target>
# For help: make help

.PHONY: help install test run plots clean docker-build docker-test docker-run docker-plots all ci-test

# Default target - display help information
help:
	@echo "SNN Log Anomaly Detection - Command Reference"
	@echo "=============================================="
	@echo ""
	@echo "Quick Start (Recommended):"
	@echo "  make docker-all          Run complete workflow in Docker (tests + pipeline + plots)"
	@echo ""
	@echo "Docker Commands:"
	@echo "  make docker-build        Build Docker image for containerized execution"
	@echo "  make docker-test         Execute test suite in Docker container"
	@echo "  make docker-run          Run full pipeline in Docker container"
	@echo "  make docker-plots        Generate thesis plots in Docker container"
	@echo ""
	@echo "Local Commands:"
	@echo "  make install             Install Python dependencies locally"
	@echo "  make test                Run unit and integration tests locally"
	@echo "  make ci-test             Run CI/CD integration test suite"
	@echo "  make run                 Execute pipeline locally"
	@echo "  make plots               Generate publication-quality plots locally"
	@echo ""
	@echo "Maintenance:"
	@echo "  make clean               Remove generated files and outputs"
	@echo "  make download-data       Download Loghub datasets"

# =============================================================================
# Local Execution Targets
# =============================================================================

# Install Python dependencies for local development
install:
	pip install -r requirements.txt
	pip install -e .

# Execute unit and integration test suite with coverage reporting
test:
	pytest tests/ -v --cov=src --cov-report=html
	@echo "Coverage report available at: htmlcov/index.html"

# Execute CI/CD integration tests for continuous integration pipeline
ci-test:
	@echo "Running CI/CD integration tests..."
	python ci_cd_test.py
	@echo "CI/CD tests completed successfully"
	@echo "Results saved to: logs/ci_cd_results.json"

# Run the main anomaly detection pipeline locally
run:
	python src/run_pipeline.py

# Generate publication-quality thesis plots from results
plots:
	python scripts/generate_thesis_plots.py

# Download Loghub datasets for evaluation
download-data:
	bash scripts/download_data.sh

# =============================================================================
# Docker Execution Targets
# =============================================================================

# Build Docker image for containerized execution
docker-build:
	docker-compose build

# Run test suite in Docker container
docker-test:
	docker-compose run --rm snn-test

# Execute full pipeline in Docker container
docker-run:
	docker-compose run --rm snn-pipeline

# Generate plots in Docker container
docker-plots:
	docker-compose run --rm snn-plots

# Execute complete workflow: build, test, run pipeline, and generate plots
docker-all: docker-build docker-test docker-run docker-plots
	@echo "Complete Docker workflow finished successfully"

# =============================================================================
# Maintenance Targets
# =============================================================================

# Remove generated files, logs, and plots
clean:
	rm -rf logs/* plots/* htmlcov .coverage .pytest_cache __pycache__
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	@echo "Cleanup completed"

# Display project information and status
info:
	@echo "Project: SNN-Based Log Anomaly Detection"
	@echo "Python Version: 3.10+"
	@echo "Dependencies: requirements.txt"
	@echo "Documentation: README.md"
