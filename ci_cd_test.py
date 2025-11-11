"""
CI/CD Automated Test Suite

This module provides comprehensive automated testing for the anomaly detection project.
It validates the complete pipeline including data loading, spike encoding, SNN models,
and baseline machine learning models using pytest framework.

Test coverage includes:
    - Data loading and CSV parsing
    - Spike encoding strategies
    - SNN forward propagation
    - Baseline model integration (Isolation Forest)
    - Result reporting in JSON format
"""

import os
import sys
import pytest
import numpy as np
import torch
from pathlib import Path
import json

from data_loader import LogDataLoader
from spike_encoder import SpikeEncoder, EncodingStrategy
from model_snn import OptimizedSpikingAnomalyDetector
from baseline_ml import IsolationForestDetector


class TestDataPipeline:
    """
    Test suite for data loading functionality.

    This class validates that log data can be properly loaded from CSV files
    and parsed into the expected format for downstream processing.
    """

    @pytest.fixture
    def sample_csv(self, tmp_path):
        """
        Create a temporary CSV file for testing.

        This fixture generates a sample CSV file with log entries containing
        LineId, Label, and EventTemplate columns.

        Args:
            tmp_path (Path): Pytest temporary directory fixture

        Returns:
            str: Path to the temporary CSV file
        """
        csv_content = """LineId,Label,EventTemplate
1,Normal,User login successful
2,Anomaly,Failed authentication attempt
3,Normal,User logout
4,Anomaly,Connection timeout
5,Normal,Session active"""

        csv_file = tmp_path / "test.csv"
        csv_file.write_text(csv_content)

        return str(csv_file)

    def test_data_loading(self, sample_csv):
        """
        Test CSV file loading and parsing.

        Validates that the LogDataLoader correctly loads a CSV file and
        returns a dataframe with the expected number of rows.

        Args:
            sample_csv (str): Path to sample CSV file from fixture
        """
        loader = LogDataLoader(sample_csv)
        df = loader.load()

        # Verify correct number of rows loaded
        assert len(df) == 5


class TestSpikeEncoding:
    """
    Test suite for spike encoding functionality.

    This class validates that numerical data can be properly encoded as
    spike trains using various encoding strategies.
    """

    def test_rate_encoding(self):
        """
        Test rate-based spike encoding.

        Validates that the rate encoding strategy produces spike trains
        with the correct shape (batch_size, time_steps, features).
        """
        encoder = SpikeEncoder(strategy=EncodingStrategy.RATE, time_steps=100)

        # Generate random test data
        data = np.random.rand(10, 20)

        # Encode as spike trains
        spike_trains = encoder.encode(data)

        # Verify output shape: (batch, time, features)
        assert spike_trains.shape == (10, 100, 20)


class TestSNNModel:
    """
    Test suite for Spiking Neural Network models.

    This class validates the SNN model architecture and ensures proper
    forward propagation through the network.
    """

    def test_snn_forward_pass(self):
        """
        Test SNN forward propagation.

        Validates that the optimized spiking anomaly detector properly
        processes input spike trains and produces classification outputs
        with the expected dimensions.
        """
        # Initialize model with test parameters
        model = OptimizedSpikingAnomalyDetector(
            input_size=20,
            hidden_size=64,
            output_size=2,
            time_steps=50
        )

        # Create random input tensor: (batch, time, features)
        x = torch.rand(4, 50, 20)

        # Forward pass through the network
        output = model(x)

        # Verify output shape: (batch, output_classes)
        assert output.shape == (4, 2)


class TestBaselineModels:
    """
    Test suite for baseline machine learning models.

    This class validates that baseline models can be properly initialized,
    trained, and used for inference.
    """

    def test_isolation_forest(self):
        """
        Test Isolation Forest model.

        Validates that the Isolation Forest detector can be trained on
        sample data and generate predictions with the correct shape.
        """
        # Initialize detector
        model = IsolationForestDetector()

        # Generate random training data
        X = np.random.rand(100, 20)

        # Train the model
        model.fit(X)

        # Generate predictions
        preds = model.predict(X)

        # Verify predictions shape matches input
        assert preds.shape == (100,)


def generate_report(exit_code):
    """
    Generate test execution report.

    Creates a JSON report containing the test execution status, exit code,
    and timestamp for CI/CD pipeline integration.

    Args:
        exit_code (int): Pytest exit code (0 for success, non-zero for failure)

    Returns:
        dict: Report dictionary containing status, exit code, and timestamp
    """
    # Ensure logs directory exists
    os.makedirs('logs', exist_ok=True)

    # Compile report data
    result = {
        'status': 'PASSED' if exit_code == 0 else 'FAILED',
        'exit_code': int(exit_code),
        'timestamp': str(Path('logs').stat().st_mtime if Path('logs').exists() else 0)
    }

    # Save report as JSON
    with open('logs/ci_cd_results.json', 'w') as f:
        json.dump(result, f, indent=2)

    return result


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("CI/CD AUTOMATED TESTS")
    print("=" * 60 + "\n")

    # Execute pytest with verbose output and short traceback
    exit_code = pytest.main([__file__, '-v', '--tb=short'])

    # Generate test report
    report = generate_report(exit_code)

    # Display final result
    print("\n" + "=" * 60)
    print(f"RESULT: {report['status']}")
    print("=" * 60 + "\n")

    # Exit with appropriate code for CI/CD systems
    sys.exit(exit_code)
