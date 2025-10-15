# ci_cd_test.py
"""
Automated CI/CD integration tests for MLOps pipeline.
Validates data loading, encoding, training, and inference steps.
"""

import os
import sys
import pytest
import numpy as np
import torch
from pathlib import Path
import json
import glob

from data_loader import LogDataLoader
from spike_encoder import SpikeEncoder, EncodingStrategy
from model_snn import SpikingLogAnomalyDetector
from baseline_ml import IsolationForestDetector, LSTMDetector


class TestDataPipeline:
    @pytest.fixture
    def sample_csv(self, tmp_path):
        csv_content = """LineId,Label,EventTemplate
1,Normal,User login
2,Anomaly,Failed attempt
3,Normal,User logout"""
        csv_file = tmp_path / "test.csv"
        csv_file.write_text(csv_content)
        return str(csv_file)

    def test_data_loading(self, sample_csv):
        loader = LogDataLoader(sample_csv)
        df = loader.load()
        assert len(df) > 0

    def test_schema_detection(self, sample_csv):
        loader = LogDataLoader(sample_csv)
        loader.load()
        schema = loader.auto_detect_schema()
        assert schema is not None


class TestSpikeEncoding:
    def test_rate_encoding(self):
        encoder = SpikeEncoder(strategy=EncodingStrategy.RATE, time_steps=100)
        data = np.random.rand(10, 20)
        spike_trains = encoder.encode(data)
        assert spike_trains.shape == (10, 20, 100)


class TestSNNModel:
    def test_snn_forward_pass(self):
        model = SpikingLogAnomalyDetector(input_size=20, hidden_size=64, time_steps=50)
        x = torch.rand(4, 50, 20)
        output = model(x)
        assert output.shape == (4, 2)


class TestBaselineModels:
    def test_isolation_forest(self):
        model = IsolationForestDetector()
        X = np.random.rand(100, 20)
        model.fit(X)
        preds = model.predict(X)
        assert preds.shape == (100,)


if __name__ == "__main__":
    os.makedirs('logs', exist_ok=True)
    exit_code = pytest.main([__file__, '-v', '--tb=short'])

    result = {'status': 'PASSED' if exit_code == 0 else 'FAILED', 'exit_code': int(exit_code)}
    with open('logs/ci_cd_results.json', 'w') as f:
        json.dump(result, f, indent=2)

    print(f"\n[CI/CD] Tests {result['status']}")
    sys.exit(exit_code)
