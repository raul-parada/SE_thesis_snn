# Makefile - Simple commands for reviewers

.PHONY: help install test run plots clean docker-build docker-test docker-run docker-plots all ci-test

help:
    @echo "SNN Log Anomaly Detection - Reviewer Commands"
    @echo "=============================================="
    @echo ""
    @echo "Quick Start (Recommended):"
    @echo "  make docker-all          - Run everything in Docker (tests + pipeline + plots)"
    @echo ""
    @echo "Docker Commands:"
    @echo "  make docker-build        - Build Docker image"
    @echo "  make docker-test         - Run tests in Docker"
    @echo "  make docker-run          - Run full pipeline in Docker"
    @echo "  make docker-plots        - Generate thesis plots in Docker"
    @echo ""
    @echo "Local Commands:"
    @echo "  make install             - Install dependencies locally"
    @echo "  make test                - Run tests locally"
    @echo "  make ci-test             - Run CI/CD integration tests"    # ← NEW
    @echo "  make run                 - Run pipeline locally"
    @echo "  make plots               - Generate plots locally"
    @echo ""
    @echo "Maintenance:"
    @echo "  make clean               - Clean generated files"
    @echo "  make download-data       - Download Loghub datasets"

# Local installation
install:
    pip install -r requirements.txt
    pip install -e .

# Run tests locally
test:
    pytest tests/ -v --cov=src --cov-report=html
    @echo "Coverage report: htmlcov/index.html"

# ✨ NEW: Run CI/CD integration tests
ci-test:
    @echo "Running CI/CD integration tests..."
    python ci_cd_test.py
    @echo "✓ CI/CD tests completed"
    @echo "Results: logs/ci_cd_results.json"

# Run pipeline locally
run:
    python src/run_pipeline.py

# Generate plots locally
plots:
    python scripts/generate_thesis_plots.py

# Download datasets
download-data:
    bash scripts/download_data.sh

# Clean generated files
clean:
    rm -rf logs/* plots
