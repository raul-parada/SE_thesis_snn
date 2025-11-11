"""
Comprehensive Test Suite for SNN-Based Log Anomaly Detection Pipeline

This module provides a complete pytest test suite covering core functionality
of the anomaly detection system including spike encoding, SNN model behavior,
utility functions, and end-to-end integration testing.

Test coverage areas:
    - Spike encoding algorithms (rate, temporal, latency)
    - SNN model correctness and gradient flow
    - Type conversion and JSON serialization utilities
    - End-to-end pipeline integration
    - Docker environment compatibility
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


# =============================================================================
# Spike Encoding Tests
# =============================================================================

class TestSpikeEncoder:
    """
    Test suite for spike encoding algorithms.

    This class validates that spike encoding produces correct output shapes,
    valid value ranges, and reasonable spike densities for different encoding
    strategies.
    """

    def test_rate_encoding_shape(self):
        """
        Test rate encoding output dimensions.

        Validates that rate encoding produces spike trains with the expected
        shape: (batch_size, time_steps, features).
        """
        data = np.random.rand(10, 20)
        encoder = SpikeEncoder(strategy=EncodingStrategy.RATE, time_steps=50)
        spike_trains = encoder.encode(data)

        # Verify correct output shape
        assert spike_trains.shape == (10, 50, 20)
        assert spike_trains.dtype == np.float32

    def test_rate_encoding_range(self):
        """
        Test rate encoding value bounds.

        Validates that all spike values are within the valid range [0, 1],
        representing spike probabilities.
        """
        data = np.random.rand(5, 10)
        encoder = SpikeEncoder(strategy=EncodingStrategy.RATE, time_steps=30)
        spike_trains = encoder.encode(data)

        # Verify all values are in valid range
        assert spike_trains.min() >= 0.0
        assert spike_trains.max() <= 1.0

    def test_temporal_encoding_shape(self):
        """
        Test temporal encoding output dimensions.

        Validates that temporal encoding produces spike trains with the expected
        shape for precise spike timing representation.
        """
        data = np.random.rand(8, 15)
        encoder = SpikeEncoder(strategy=EncodingStrategy.TEMPORAL, time_steps=40)
        spike_trains = encoder.encode(data)

        # Verify correct output shape
        assert spike_trains.shape == (8, 40, 15)

    def test_spike_density_reasonable(self):
        """
        Test spike density within expected bounds.

        Validates that spike density is reasonable for the specified max_rate,
        ensuring energy-efficient sparse encoding.
        """
        data = np.random.rand(10, 20)
        encoder = SpikeEncoder(strategy=EncodingStrategy.RATE, time_steps=50, max_rate=0.5)
        spike_trains = encoder.encode(data)
        density = spike_trains.mean()

        # Verify density is within reasonable range
        assert 0 < density < 0.5


# =============================================================================
# SNN Model Tests
# =============================================================================

class TestSNNModel:
    """
    Test suite for Spiking Neural Network model.

    This class validates model initialization, forward pass correctness,
    gradient flow, and numerical stability.
    """

    @pytest.fixture
    def sample_model(self):
        """
        Create a small SNN model for testing.

        Returns:
            OptimizedSpikingAnomalyDetector: Initialized SNN model
        """
        return OptimizedSpikingAnomalyDetector(
            input_size=20,
            hidden_size=32,
            output_size=2,
            time_steps=50,
            sparsity=0.3
        )

    def test_model_initialization(self, sample_model):
        """
        Test model initialization and parameter setup.

        Validates that the model initializes with correct architecture
        parameters.
        """
        assert sample_model.input_size == 20
        assert sample_model.hidden_size == 32
        assert sample_model.output_size == 2

    def test_forward_pass_shape(self, sample_model):
        """
        Test forward pass output dimensions.

        Validates that the model produces output with the expected shape
        for binary classification.
        """
        batch_size = 8
        x = torch.rand(batch_size, 50, 20)
        output = sample_model(x)

        # Verify output shape matches expected classification dimensions
        assert output.shape == (batch_size, 2)

    def test_forward_pass_no_nans(self, sample_model):
        """
        Test numerical stability of forward pass.

        Validates that the model does not produce NaN or infinite values,
        ensuring numerical stability during inference.
        """
        x = torch.rand(5, 50, 20)
        output = sample_model(x)

        # Verify no numerical issues
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()

    def test_gradient_flow(self, sample_model):
        """
        Test gradient propagation through the network.

        Validates that gradients flow correctly through all layers,
        ensuring the model can learn during training.
        """
        x = torch.rand(3, 50, 20)
        target = torch.tensor([0, 1, 0], dtype=torch.long)

        # Perform forward and backward pass
        output = sample_model(x)
        loss = torch.nn.functional.cross_entropy(output, target)
        loss.backward()

        # Check that at least some gradients are non-zero
        has_grad = any(p.grad is not None and p.grad.abs().sum() > 0
                       for p in sample_model.parameters())
        assert has_grad


# =============================================================================
# Utility Function Tests
# =============================================================================

class TestUtilities:
    """
    Test suite for utility functions.

    This class validates type conversion and JSON serialization utilities
    used for logging and result storage.
    """

    def test_convert_numpy_to_native(self):
        """
        Test NumPy to Python native type conversion.

        Validates that NumPy types are correctly converted to Python native
        types for JSON serialization.
        """
        data = {
            'int64': np.int64(42),
            'float64': np.float64(3.14),
            'array': np.array([1, 2, 3]),
            'bool': np.bool_(True)
        }

        converted = convert_to_native_types(data)

        # Verify all types are converted to Python natives
        assert isinstance(converted['int64'], int)
        assert isinstance(converted['float64'], float)
        assert isinstance(converted['array'], list)
        assert isinstance(converted['bool'], bool)

    def test_json_serialization(self):
        """
        Test JSON serialization of converted data.

        Validates that converted data can be successfully serialized to JSON
        and deserialized back to Python objects.
        """
        data = {
            'accuracy': np.float64(0.95),
            'loss': np.float32(0.12),
            'params': np.int64(11076)
        }

        converted = convert_to_native_types(data)
        json_str = json.dumps(converted)

        # Verify JSON serialization and deserialization
        assert isinstance(json_str, str)
        parsed = json.loads(json_str)
        assert parsed['accuracy'] == 0.95


# =============================================================================
# Integration Tests
# =============================================================================

class TestIntegration:
    """
    End-to-end integration test suite.

    This class validates that the complete pipeline works correctly from
    data input through encoding, model inference, and loss calculation.
    """

    def test_end_to_end_pipeline(self):
        """
        Test complete pipeline with synthetic data.

        This integration test validates that all pipeline components work
        together correctly: data generation, spike encoding, model inference,
        and loss computation.
        """
        # Step 1: Generate synthetic input data
        num_samples = 50
        feature_dim = 20
        time_steps = 30
        synthetic_data = np.random.rand(num_samples, feature_dim)
        synthetic_labels = np.random.randint(0, 2, num_samples)

        # Step 2: Encode data as spike trains
        encoder = SpikeEncoder(strategy=EncodingStrategy.RATE, time_steps=time_steps)
        spike_trains = encoder.encode(synthetic_data)
        assert spike_trains.shape == (num_samples, time_steps, feature_dim)

        # Step 3: Create SNN model
        model = OptimizedSpikingAnomalyDetector(
            input_size=feature_dim,
            hidden_size=32,
            output_size=2,
            time_steps=time_steps
        )

        # Step 4: Run forward pass and compute loss
        X = torch.tensor(spike_trains, dtype=torch.float32)
        y = torch.tensor(synthetic_labels, dtype=torch.long)
        output = model(X)
        loss = torch.nn.functional.cross_entropy(output, y)

        # Step 5: Validate pipeline completes successfully
        assert output.shape == (num_samples, 2)
        assert loss.item() > 0
        assert not torch.isnan(loss)


# =============================================================================
# Docker Environment Tests
# =============================================================================

@pytest.mark.docker
class TestDockerEnvironment:
    """
    Docker environment validation test suite.

    This class validates that the Docker environment is correctly configured
    with all required dependencies and permissions.
    """

    def test_temp_directory_writable(self, tmp_path):
        """
        Test temporary directory write permissions.

        Validates that the application can create and write to temporary
        directories, which is necessary for logging and output generation.
        """
        test_file = tmp_path / "test.txt"
        test_file.write_text("Docker test")

        # Verify file creation and content
        assert test_file.exists()
        assert test_file.read_text() == "Docker test"

    def test_pytorch_available(self):
        """
        Test PyTorch availability and functionality.

        Validates that PyTorch is installed and can perform basic tensor
        operations.
        """
        x = torch.tensor([1.0, 2.0, 3.0])
        assert x.sum() == 6.0

    def test_numpy_available(self):
        """
        Test NumPy availability and functionality.

        Validates that NumPy is installed and can perform basic array
        operations.
        """
        arr = np.array([1, 2, 3])
        assert arr.sum() == 6


# =============================================================================
# Test Execution
# =============================================================================

if __name__ == "__main__":
    """
    Direct test execution entry point.

    Allows running the test suite directly with verbose output and
    short traceback format.
    """
    pytest.main([__file__, '-v', '--tb=short'])
