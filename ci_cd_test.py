# ci_cd_test.py
"""
Minimal CI/CD tests for thesis project.
Validates data loading, encoding, models, and basic pipeline.
"""

import os
import sys
import pytest
import numpy as np
import torch
from pathlib import Path
import json

# Import your existing modules
from data_loader import LogDataLoader
from spike_encoder import SpikeEncoder, EncodingStrategy
from model_snn import OptimizedSpikingAnomalyDetector
from baseline_ml import IsolationForestDetector


class TestDataPipeline:
    """Test data loading."""

    @pytest.fixture
    def sample_csv(self, tmp_path):
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
        """Test CSV loading."""
        loader = LogDataLoader(sample_csv)
        df = loader.load()
        assert len(df) == 5


class TestSpikeEncoding:
    """Test spike encoding."""

    def test_rate_encoding(self):
        """Test rate encoding."""
        encoder = SpikeEncoder(strategy=EncodingStrategy.RATE, time_steps=100)
        data = np.random.rand(10, 20)
        spike_trains = encoder.encode(data)
        assert spike_trains.shape == (10, 100, 20)  # Note: your encoder uses (batch, time, features)


class TestSNNModel:
    """Test SNN model."""

    def test_snn_forward_pass(self):
        """Test SNN forward propagation."""
        model = OptimizedSpikingAnomalyDetector(
            input_size=20,
            hidden_size=64,
            output_size=2,
            time_steps=50
        )
        x = torch.rand(4, 50, 20)
        output = model(x)
        assert output.shape == (4, 2)


class TestBaselineModels:
    """Test baseline models."""

    def test_isolation_forest(self):
        """Test IsolationForest."""
        model = IsolationForestDetector()
        X = np.random.rand(100, 20)
        model.fit(X)
        preds = model.predict(X)
        assert preds.shape == (100,)


def generate_report(exit_code):
    """Generate test report."""
    os.makedirs('logs', exist_ok=True)

    result = {
        'status': 'PASSED' if exit_code == 0 else 'FAILED',
        'exit_code': int(exit_code),
        'timestamp': str(Path('logs').stat().st_mtime if Path('logs').exists() else 0)
    }

    with open('logs/ci_cd_results.json', 'w') as f:
        json.dump(result, f, indent=2)

    return result


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("CI/CD AUTOMATED TESTS")
    print("=" * 60 + "\n")

    # Run pytest
    exit_code = pytest.main([__file__, '-v', '--tb=short'])

    # Generate report
    report = generate_report(exit_code)

    print("\n" + "=" * 60)
    print(f"RESULT: {report['status']}")
    print("=" * 60 + "\n")

    sys.exit(exit_code)
