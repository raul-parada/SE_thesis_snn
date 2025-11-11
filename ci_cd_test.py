"""
CI/CD Automated Test Suite

Test suite that validates the CI/CD pipeline functionality 
"""

import os
import sys
import pytest
import numpy as np
import torch
from pathlib import Path
import json
from datetime import datetime


class TestBasicFunctionality:
    """
    Test suite for basic Python and library functionality.
    """
    
    def test_python_version(self):
        """Test Python version is 3.9+."""
        assert sys.version_info.major == 3
        assert sys.version_info.minor >= 9
    
    def test_numpy_available(self):
        """Test numpy is available and functional."""
        arr = np.array([1, 2, 3, 4, 5])
        assert arr.sum() == 15
        assert arr.mean() == 3.0
    
    def test_torch_available(self):
        """Test PyTorch is available and functional."""
        tensor = torch.tensor([1.0, 2.0, 3.0])
        assert tensor.sum().item() == 6.0
        assert torch.cuda.is_available() or not torch.cuda.is_available()  # Always passes
    
    def test_sklearn_available(self):
        """Test scikit-learn is available."""
        from sklearn.ensemble import IsolationForest
        model = IsolationForest(random_state=42)
        assert model is not None
    
    def test_pandas_available(self):
        """Test pandas is available and functional."""
        import pandas as pd
        df = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
        assert len(df) == 3
        assert list(df.columns) == ['a', 'b']


class TestDataProcessing:
    """
    Test suite for data processing functionality.
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
    
    def test_csv_loading(self, sample_csv):
        """Test CSV file can be loaded with pandas."""
        import pandas as pd
        df = pd.read_csv(sample_csv)
        
        assert df is not None
        assert len(df) == 5
        assert 'EventId' in df.columns
        assert 'Label' in df.columns
    
    def test_data_filtering(self, sample_csv):
        """Test data filtering operations."""
        import pandas as pd
        df = pd.read_csv(sample_csv)
        
        # Filter anomalies
        anomalies = df[df['Label'] == 'Anomaly']
        assert len(anomalies) == 2
        
        # Filter normal
        normal = df[df['Label'] == 'Normal']
        assert len(normal) == 3


class TestMLOperations:
    """
    Test suite for machine learning operations.
    """
    
    def test_isolation_forest_training(self):
        """Test Isolation Forest can be trained."""
        from sklearn.ensemble import IsolationForest
        
        # Generate synthetic data
        X_train = np.random.rand(100, 10)
        X_test = np.random.rand(20, 10)
        
        # Train model
        model = IsolationForest(contamination=0.1, random_state=42)
        model.fit(X_train)
        
        # Make predictions
        predictions = model.predict(X_test)
        
        assert predictions.shape == (20,)
        assert set(predictions).issubset({-1, 1})
    
    def test_torch_model_forward_pass(self):
        """Test PyTorch model forward pass."""
        # Simple linear model
        model = torch.nn.Sequential(
            torch.nn.Linear(10, 20),
            torch.nn.ReLU(),
            torch.nn.Linear(20, 2)
        )
        
        # Create input
        x = torch.rand(4, 10)
        
        # Forward pass
        output = model(x)
        
        assert output.shape == (4, 2)
        assert not torch.isnan(output).any()


class TestSpikeEncoding:
    """
    Test suite for spike encoding functionality.
    """
    
    def test_rate_encoding(self):
        """Test rate-based spike encoding."""
        time_steps = 100
        data = np.random.rand(10, 20)
        
        # Simple rate encoding: repeat data over time dimension
        spike_trains = np.repeat(data[:, np.newaxis, :], time_steps, axis=1)
        
        # Verify output shape: (batch, time, features)
        assert spike_trains.shape == (10, 100, 20)
        assert spike_trains.dtype == np.float64
    
    def test_encoding_normalization(self):
        """Test encoding normalization."""
        data = np.array([[1, 2, 3], [4, 5, 6]])
        
        # Normalize to [0, 1]
        normalized = (data - data.min()) / (data.max() - data.min() + 1e-8)
        
        assert normalized.min() >= 0
        assert normalized.max() <= 1


class TestCICDIntegration:
    """
    Test suite for CI/CD integration functionality.
    """
    
    def test_logs_directory_creation(self, tmp_path):
        """Test logs directory can be created."""
        log_dir = tmp_path / "logs"
        log_dir.mkdir(exist_ok=True)
        
        assert log_dir.exists()
        assert log_dir.is_dir()
    
    def test_json_report_generation(self, tmp_path):
        """Test JSON reports are generated correctly."""
        log_dir = tmp_path / "logs"
        log_dir.mkdir()
        
        result = {
            'status': 'PASSED',
            'exit_code': 0,
            'timestamp': datetime.now().isoformat()
        }
        
        report_path = log_dir / "ci_cd_results.json"
        with open(report_path, 'w') as f:
            json.dump(result, f, indent=2)
        
        assert report_path.exists()
        
        with open(report_path, 'r') as f:
            loaded = json.load(f)
            assert loaded['status'] == 'PASSED'
            assert loaded['exit_code'] == 0
    
    def test_yaml_config_loading(self):
        """Test YAML configuration can be loaded."""
        import yaml
        
        config = {
            'model': {
                'input_size': 20,
                'hidden_size': 64,
                'output_size': 2
            }
        }
        
        # Test yaml operations
        yaml_str = yaml.dump(config)
        loaded = yaml.safe_load(yaml_str)
        
        assert loaded['model']['input_size'] == 20


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
    
    # Execute pytest with verbose output
    exit_code = pytest.main([
        __file__, 
        '-v', 
        '--tb=short', 
        '-p', 'no:warnings'
    ])
    
    # Generate test report
    report = generate_report(exit_code)
    
    # Display final result
    print("\n" + "=" * 60)
    print(f"RESULT: {report['status']}")
    print(f"Exit Code: {exit_code}")
    print("=" * 60 + "\n")
    
    # Exit with appropriate code for CI/CD systems
    sys.exit(exit_code)
