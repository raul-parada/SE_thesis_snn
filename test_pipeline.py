# test_pipeline.py
"""
Minimal pytest suite for generalized SNN Log Anomaly Detection pipeline.
Tests core functionality without dataset-specific assumptions.
"""

import pytest
import numpy as np
import torch
import tempfile
import json
from pathlib import Path

from spike_encoder import SpikeEncoder, EncodingStrategy
from model_snn import OptimizedSpikingAnomalyDetector
from evaluation import convert_to_native_types


# ============================================================================
# 1. SPIKE ENCODING TESTS (Core Algorithm)
# ============================================================================

class TestSpikeEncoder:
    """Test spike encoding logic (dataset-agnostic)."""

    def test_rate_encoding_shape(self):
        """Test rate encoding produces correct output shape."""
        data = np.random.rand(10, 20)
        encoder = SpikeEncoder(strategy=EncodingStrategy.RATE, time_steps=50)
        spike_trains = encoder.encode(data)

        assert spike_trains.shape == (10, 50, 20)  # (batch, time_steps, features)
        assert spike_trains.dtype == np.float32

    def test_rate_encoding_range(self):
        """Test spike values are in valid range [0,1]."""
        data = np.random.rand(5, 10)
        encoder = SpikeEncoder(strategy=EncodingStrategy.RATE, time_steps=30)
        spike_trains = encoder.encode(data)

        assert spike_trains.min() >= 0.0
        assert spike_trains.max() <= 1.0

    def test_temporal_encoding_shape(self):
        """Test temporal encoding produces correct output shape."""
        data = np.random.rand(8, 15)
        encoder = SpikeEncoder(strategy=EncodingStrategy.TEMPORAL, time_steps=40)
        spike_trains = encoder.encode(data)

        assert spike_trains.shape == (8, 40, 15)  # (batch, time_steps, features)

    def test_spike_density_reasonable(self):
        """Test spike density is within reasonable range."""
        data = np.random.rand(10, 20)
        encoder = SpikeEncoder(strategy=EncodingStrategy.RATE, time_steps=50, max_rate=0.5)
        spike_trains = encoder.encode(data)

        density = spike_trains.mean()
        assert 0 < density < 0.5


# ============================================================================
# 2. SNN MODEL TESTS (Neural Network Correctness)
# ============================================================================

class TestSNNModel:
    """Test SNN model behavior (architecture-agnostic)."""

    @pytest.fixture
    def sample_model(self):
        """Create a small SNN model for testing."""
        return OptimizedSpikingAnomalyDetector(
            input_size=20,
            hidden_size=32,
            output_size=2,
            time_steps=50,
            sparsity=0.3
        )

    def test_model_initialization(self, sample_model):
        """Test model initializes without errors."""
        assert sample_model.input_size == 20
        assert sample_model.hidden_size == 32
        assert sample_model.output_size == 2

    def test_forward_pass_shape(self, sample_model):
        """Test forward pass produces correct output shape."""
        batch_size = 8
        x = torch.rand(batch_size, 50, 20)  # (batch, time_steps, features)

        output = sample_model(x)

        assert output.shape == (batch_size, 2)

    def test_forward_pass_no_nans(self, sample_model):
        """Test forward pass doesn't produce NaN values."""
        x = torch.rand(5, 50, 20)
        output = sample_model(x)

        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()

    def test_gradient_flow(self, sample_model):
        """Test gradients flow through the model."""
        x = torch.rand(3, 50, 20)
        target = torch.tensor([0, 1, 0], dtype=torch.long)

        output = sample_model(x)
        loss = torch.nn.functional.cross_entropy(output, target)
        loss.backward()

        # Check at least some gradients are non-zero
        has_grad = any(p.grad is not None and p.grad.abs().sum() > 0
                       for p in sample_model.parameters())
        assert has_grad


# ============================================================================
# 3. UTILITY FUNCTION TESTS (Data Handling)
# ============================================================================

class TestUtilities:
    """Test utility functions (JSON serialization, type conversion)."""

    def test_convert_numpy_to_native(self):
        """Test numpy to Python type conversion."""
        data = {
            'int64': np.int64(42),
            'float64': np.float64(3.14),
            'array': np.array([1, 2, 3]),
            'bool': np.bool_(True)
        }

        converted = convert_to_native_types(data)

        assert isinstance(converted['int64'], int)
        assert isinstance(converted['float64'], float)
        assert isinstance(converted['array'], list)
        assert isinstance(converted['bool'], bool)

    def test_json_serialization(self):
        """Test data can be serialized to JSON (for logging/output)."""
        data = {
            'accuracy': np.float64(0.95),
            'loss': np.float32(0.12),
            'params': np.int64(11076)
        }

        converted = convert_to_native_types(data)
        json_str = json.dumps(converted)

        assert isinstance(json_str, str)
        parsed = json.loads(json_str)
        assert parsed['accuracy'] == 0.95


# ============================================================================
# 4. INTEGRATION TEST (End-to-End Smoke Test)
# ============================================================================

class TestIntegration:
    """Minimal integration test (proves pipeline works end-to-end)."""

    def test_end_to_end_pipeline(self):
        """Test complete pipeline with synthetic data."""
        # 1. Generate synthetic input
        num_samples = 50
        feature_dim = 20
        time_steps = 30

        synthetic_data = np.random.rand(num_samples, feature_dim)
        synthetic_labels = np.random.randint(0, 2, num_samples)

        # 2. Encode to spikes
        encoder = SpikeEncoder(strategy=EncodingStrategy.RATE, time_steps=time_steps)
        spike_trains = encoder.encode(synthetic_data)

        assert spike_trains.shape == (num_samples, time_steps, feature_dim)

        # 3. Create model
        model = OptimizedSpikingAnomalyDetector(
            input_size=feature_dim,
            hidden_size=32,
            output_size=2,
            time_steps=time_steps
        )

        # 4. Run forward pass
        X = torch.tensor(spike_trains, dtype=torch.float32)
        y = torch.tensor(synthetic_labels, dtype=torch.long)

        output = model(X)
        loss = torch.nn.functional.cross_entropy(output, y)

        # 5. Validate pipeline completes
        assert output.shape == (num_samples, 2)
        assert loss.item() > 0
        assert not torch.isnan(loss)


# ============================================================================
# 5. DOCKER COMPATIBILITY TESTS (Environment Validation)
# ============================================================================

@pytest.mark.docker
class TestDockerEnvironment:
    """Test Docker environment setup is correct."""

    def test_temp_directory_writable(self, tmp_path):
        """Test temporary directories can be created (for outputs)."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("Docker test")

        assert test_file.exists()
        assert test_file.read_text() == "Docker test"

    def test_pytorch_available(self):
        """Test PyTorch is installed and working."""
        x = torch.tensor([1.0, 2.0, 3.0])
        assert x.sum() == 6.0

    def test_numpy_available(self):
        """Test NumPy is installed and working."""
        arr = np.array([1, 2, 3])
        assert arr.sum() == 6


# ============================================================================
# PYTEST CONFIGURATION
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, '-v', '--tb=short'])
