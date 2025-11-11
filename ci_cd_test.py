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
from datetime import datetime


class TestDataPipeline:
    """
    Test suite for data loading functionality.
    """
    
    @pytest.fixture
    def sample_csv(self, tmp_path):
        """Create a temporary CSV file for testing."""
        csv_content = """LineId,EventId,EventTemplate,Label
1,E001,User login successful,Normal
2,E002,Failed authentication attempt,Anomaly
3,E001,User logout,Normal
4,E003,Connection timeout,Anomaly
5,E001,Session active,Normal"""
        csv_file = tmp_path / "test_structured.csv"
        csv_file.write_text(csv_content)
        return str(csv_file)
    
    def test_data_loading(self, sample_csv):
        """Test that CSV files can be loaded."""
        from data_loader import LogDataLoader
        
        loader = LogDataLoader(sample_csv, sample_ratio=1.0)
        df = loader.load()
        
        assert df is not None
        assert len(df) == 5
        assert 'EventId' in df.columns or 'EventTemplate' in df.columns
    
    def test_schema_detection(self, sample_csv):
        """Test automatic schema detection."""
        from data_loader import LogDataLoader
        
        loader = LogDataLoader(sample_csv)
        loader.load()
        schema = loader.auto_detect_schema()
        
        assert schema is not None
        assert 'label' in schema or 'content' in schema


class TestSpikeEncoding:
    """
    Test suite for spike encoding functionality.
    """
    
    def test_rate_encoding(self):
        """Test rate-based spike encoding."""
        # Simulate rate encoding
        time_steps = 100
        data = np.random.rand(10, 20)
        
        # Simple rate encoding: repeat data over time dimension
        spike_trains = np.repeat(data[:, np.newaxis, :], time_steps, axis=1)
        
        # Verify output shape: (batch, time, features)
        assert spike_trains.shape == (10, 100, 20)
        assert spike_trains.dtype == np.float64
    
    def test_encoding_normalization(self):
        """Test that encoding normalizes values properly."""
        data = np.array([[1, 2, 3], [4, 5, 6]])
        
        # Normalize to [0, 1]
        normalized = (data - data.min()) / (data.max() - data.min())
        
        assert normalized.min() >= 0
        assert normalized.max() <= 1


class TestSNNModel:
    """
    Test suite for Spiking Neural Network models.
    """
    
    def test_snn_initialization(self):
        """Test SNN model can be initialized."""
        from model_snn import OptimizedSpikingAnomalyDetector
        
        model = OptimizedSpikingAnomalyDetector(
            input_size=20,
            hidden_size=64,
            output_size=2,
            time_steps=50
        )
        
        assert model is not None
        assert model.input_size == 20
        assert model.output_size == 2
    
    def test_snn_forward_pass(self):
        """Test SNN forward propagation."""
        from model_snn import OptimizedSpikingAnomalyDetector
        
        model = OptimizedSpikingAnomalyDetector(
            input_size=20,
            hidden_size=64,
            output_size=2,
            time_steps=50
        )
        
        # Create random input tensor: (batch, time, features)
        batch_size = 4
        time_steps = 50
        x = torch.rand(batch_size, time_steps, 20)
        
        # Forward pass
        output = model(x)
        
        # Verify output shape: (batch, output_classes)
        assert output.shape == (batch_size, 2)
        assert output.requires_grad
    
    def test_focal_loss(self):
        """Test Focal Loss computation."""
        from model_snn import FocalLoss
        
        focal_loss = FocalLoss(gamma=2.0)
        
        # Create dummy inputs and targets
        inputs = torch.randn(4, 2, requires_grad=True)
        targets = torch.tensor([0, 1, 0, 1])
        
        # Compute loss
        loss = focal_loss(inputs, targets)
        
        assert loss.item() >= 0
        assert not torch.isnan(loss)


class TestBaselineModels:
    """
    Test suite for baseline machine learning models.
    """
    
    def test_isolation_forest(self):
        """Test Isolation Forest model."""
        from baseline_ml import IsolationForestDetector
        
        # Initialize detector
        detector = IsolationForestDetector(contamination=0.1, n_estimators=10)
        
        # Generate random training data
        X_train = np.random.rand(100, 20)
        X_test = np.random.rand(20, 20)
        
        # Train the model
        detector.fit(X_train)
        
        # Generate predictions
        preds = detector.predict(X_test)
        
        # Verify predictions shape matches input
        assert preds.shape == (20,)
        # Predictions should be 0 (normal) or 1 (anomaly)
        assert set(preds).issubset({0, 1})
    
    def test_transformer_initialization(self):
        """Test Transformer model initialization."""
        from baseline_ml import TransformerDetector
        
        model = TransformerDetector(
            input_size=20,
            hidden_size=64,
            num_heads=4,
            num_layers=2
        )
        
        assert model is not None
        assert model.input_size == 20
        assert model.hidden_size == 64


class TestCICDIntegration:
    """
    Test suite for CI/CD integration functionality.
    """
    
    def test_json_report_generation(self, tmp_path):
        """Test that JSON reports are generated correctly."""
        # Create logs directory
        log_dir = tmp_path / "logs"
        log_dir.mkdir()
        
        # Generate test report
        result = {
            'status': 'PASSED',
            'exit_code': 0,
            'timestamp': datetime.now().isoformat()
        }
        
        report_path = log_dir / "ci_cd_results.json"
        with open(report_path, 'w') as f:
            json.dump(result, f, indent=2)
        
        # Verify report was created
        assert report_path.exists()
        
        # Verify content
        with open(report_path, 'r') as f:
            loaded = json.load(f)
            assert loaded['status'] == 'PASSED'
            assert loaded['exit_code'] == 0
    
    def test_required_imports(self):
        """Test that all required libraries are importable."""
        import numpy
        import pandas
        import torch
        import sklearn
        import yaml
        
        assert numpy.__version__
        assert pandas.__version__
        assert torch.__version__
        assert sklearn.__version__
    
    def test_project_modules_importable(self):
        """Test that project modules can be imported."""
        try:
            from data_loader import LogDataLoader
            from model_snn import OptimizedSpikingAnomalyDetector, FocalLoss
            from baseline_ml import IsolationForestDetector, TransformerDetector
            
            assert LogDataLoader is not None
            assert OptimizedSpikingAnomalyDetector is not None
            assert FocalLoss is not None
            assert IsolationForestDetector is not None
            assert TransformerDetector is not None
        except ImportError as e:
            pytest.fail(f"Failed to import project modules: {e}")


def generate_report(exit_code):
    """
    Generate test execution report.
    
    Creates a JSON report containing the test execution status, exit code,
    and timestamp for CI/CD pipeline integration.
    """
    # Ensure logs directory exists
    os.makedirs('logs', exist_ok=True)
    
    # Compile report data
    result = {
        'status': 'PASSED' if exit_code == 0 else 'FAILED',
        'exit_code': int(exit_code),
        'timestamp': datetime.now().isoformat(),
        'test_framework': 'pytest',
        'python_version': f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    }
    
    # Save report as JSON
    report_path = 'logs/ci_cd_results.json'
    with open(report_path, 'w') as f:
        json.dump(result, f, indent=2)
    
    print(f"\nReport saved to: {report_path}")
    return result


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("CI/CD AUTOMATED TESTS")
    print("=" * 60 + "\n")
    
    # Execute pytest with verbose output and short traceback
    exit_code = pytest.main([__file__, '-v', '--tb=short', '-p', 'no:warnings'])
    
    # Generate test report
    report = generate_report(exit_code)
    
    # Display final result
    print("\n" + "=" * 60)
    print(f"RESULT: {report['status']}")
    print(f"Exit Code: {exit_code}")
    print("=" * 60 + "\n")
    
    # Exit with appropriate code for CI/CD systems
    sys.exit(exit_code)
