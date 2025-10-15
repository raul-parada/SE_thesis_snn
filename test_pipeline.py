# test_pipeline.py
"""
Comprehensive pytest suite for SNN Log Anomaly Detection pipeline.
Tests all components: data loading, encoding, models, evaluation.
"""

import pytest
import numpy as np
import torch
import tempfile
import json
from pathlib import Path
import pandas as pd

from data_loader import LogDataLoader
from spike_encoder import SpikeEncoder, EncodingStrategy
from model_snn import OptimizedSpikingAnomalyDetector, SNNTrainer
from baseline_ml import IsolationForestDetector, TransformerDetector, TransformerTrainer
from evaluation import EngineeringMetricsEvaluator, convert_to_native_types


class TestDataLoader:
    """Test suite for data loading and preprocessing."""

    @pytest.fixture
    def sample_csv(self, tmp_path):
        """Create a sample CSV file for testing."""
        csv_content = """LineId,Level,Content,EventTemplate
1,INFO,User logged in successfully,User logged in
2,WARN,Connection timeout detected,Connection timeout
3,INFO,Processing request,Processing request
4,ERROR,Failed to authenticate user,Failed to authenticate
5,INFO,Request completed,Request completed"""

        csv_file = tmp_path / "test_log.csv"
        csv_file.write_text(csv_content)
        return str(csv_file)

    def test_load_csv(self, sample_csv):
        """Test CSV loading functionality."""
        loader = LogDataLoader(sample_csv)
        df = loader.load()

        assert df is not None
        assert len(df) == 5
        assert 'Level' in df.columns
        assert 'Content' in df.columns

    def test_schema_detection(self, sample_csv):
        """Test automatic schema detection."""
        loader = LogDataLoader(sample_csv)
        loader.load()
        schema = loader.auto_detect_schema()

        assert schema is not None
        assert schema['label'] == 'Level'
        assert schema['content'] == 'Content'

    def test_label_normalization(self, sample_csv):
        """Test severity level normalization."""
        loader = LogDataLoader(sample_csv)
        loader.load()
        loader.auto_detect_schema()

        normalized = loader.normalize_labels()

        assert normalized is not None
        assert len(normalized) == 5
        # WARN and ERROR should be labeled as anomalies (1)
        assert normalized[1] == 1  # WARN
        assert normalized[3] == 1  # ERROR
        # INFO should be normal (0)
        assert normalized[0] == 0

    def test_sequence_generation(self, sample_csv):
        """Test sequence generation with sliding window."""
        loader = LogDataLoader(sample_csv)
        loader.load()
        loader.auto_detect_schema()

        sequences, labels = loader.get_sequences(window_size=2, stride=1)

        assert len(sequences) > 0
        assert len(labels) == len(sequences)

    def test_tokenization(self, sample_csv):
        """Test content tokenization."""
        loader = LogDataLoader(sample_csv)
        loader.load()
        loader.auto_detect_schema()

        tokenized, vocab = loader.tokenize_content()

        assert len(tokenized) == 5
        assert len(vocab) > 0
        assert '<UNK>' in vocab


class TestSpikeEncoder:
    """Test suite for spike encoding."""

    @pytest.fixture
    def sample_data(self):
        """Create sample data for encoding."""
        return np.random.rand(10, 20)

    def test_rate_encoding(self, sample_data):
        """Test rate coding strategy."""
        encoder = SpikeEncoder(strategy=EncodingStrategy.RATE, time_steps=50)
        spike_trains = encoder.encode(sample_data)

        assert spike_trains.shape == (10, 20, 50)
        assert spike_trains.dtype == np.float32
        assert 0 <= spike_trains.max() <= 1

    def test_temporal_encoding(self, sample_data):
        """Test temporal coding strategy."""
        encoder = SpikeEncoder(strategy=EncodingStrategy.TEMPORAL, time_steps=50)
        spike_trains = encoder.encode(sample_data)

        assert spike_trains.shape == (10, 20, 50)
        assert spike_trains.dtype == np.float32

    def test_spike_density(self, sample_data):
        """Test spike density is within reasonable range."""
        encoder = SpikeEncoder(strategy=EncodingStrategy.RATE, time_steps=50, max_rate=0.5)
        spike_trains = encoder.encode(sample_data)

        density = spike_trains.mean()
        assert 0 < density < 0.5  # Should be less than max_rate


class TestSNNModel:
    """Test suite for SNN model."""

    @pytest.fixture
    def sample_model(self):
        """Create a sample SNN model."""
        return OptimizedSpikingAnomalyDetector(
            input_size=20,
            hidden_size=64,
            output_size=2,
            time_steps=50,
            sparsity=0.3
        )

    def test_model_initialization(self, sample_model):
        """Test model initializes correctly."""
        assert sample_model.input_size == 20
        assert sample_model.hidden_size == 64
        assert sample_model.output_size == 2
        assert sample_model.effective_timesteps == 10  # 50 / 5

    def test_forward_pass(self, sample_model):
        """Test forward pass with sample input."""
        batch_size = 5
        x = torch.rand(batch_size, 50, 20)

        output = sample_model(x)

        assert output.shape == (batch_size, 2)
        assert not torch.isnan(output).any()

    def test_parameter_count(self, sample_model):
        """Test model has expected number of parameters."""
        total_params = sum(p.numel() for p in sample_model.parameters())

        # Should have ~11K params with these settings
        assert 8000 < total_params < 15000

    def test_energy_proxy(self, sample_model):
        """Test energy proxy calculation."""
        x = torch.rand(5, 50, 20)
        _ = sample_model(x)

        energy = sample_model.get_energy_proxy()

        assert 0 <= energy <= 10  # Reasonable range


class TestTransformerModel:
    """Test suite for Transformer baseline."""

    @pytest.fixture
    def sample_transformer(self):
        """Create a sample Transformer model."""
        return TransformerDetector(
            input_size=20,
            hidden_size=64,
            num_heads=4,
            num_layers=2,
            output_size=2
        )

    def test_transformer_initialization(self, sample_transformer):
        """Test Transformer initializes correctly."""
        assert sample_transformer.input_size == 20
        assert sample_transformer.hidden_size == 64

    def test_transformer_forward(self, sample_transformer):
        """Test Transformer forward pass."""
        x = torch.rand(5, 1, 20)  # (batch, seq_len, features)

        output = sample_transformer(x)

        assert output.shape == (5, 2)
        assert not torch.isnan(output).any()


class TestIsolationForest:
    """Test suite for Isolation Forest baseline."""

    def test_fit_predict(self):
        """Test Isolation Forest training and prediction."""
        X_train = np.random.rand(100, 20)
        X_test = np.random.rand(20, 20)
        y_test = np.random.randint(0, 2, 20)

        model = IsolationForestDetector(track_emissions=False)
        model.fit(X_train)

        predictions = model.predict(X_test)

        assert predictions.shape == (20,)
        assert set(predictions).issubset({0, 1})

    def test_evaluate(self):
        """Test evaluation metrics."""
        X = np.random.rand(50, 20)
        y = np.random.randint(0, 2, 50)

        model = IsolationForestDetector(track_emissions=False)
        model.fit(X[:30])

        metrics = model.evaluate(X[30:], y[30:])

        assert 'accuracy' in metrics
        assert 'f1' in metrics
        assert 'latency_ms' in metrics


class TestEvaluation:
    """Test suite for evaluation metrics."""

    def test_convert_to_native_types(self):
        """Test numpy to Python type conversion."""
        data = {
            'int64': np.int64(42),
            'float64': np.float64(3.14),
            'array': np.array([1, 2, 3]),
            'nested': {
                'value': np.int32(10)
            }
        }

        converted = convert_to_native_types(data)

        assert isinstance(converted['int64'], int)
        assert isinstance(converted['float64'], float)
        assert isinstance(converted['array'], list)
        assert isinstance(converted['nested']['value'], int)

    def test_engineering_metrics(self, tmp_path):
        """Test software engineering metrics calculation."""
        # Create a dummy Python file
        test_file = tmp_path / "test_code.py"
        test_file.write_text("""
def simple_function():
    x = 1
    return x

def complex_function(a, b):
    if a > b:
        return a
    else:
        return b
""")

        evaluator = EngineeringMetricsEvaluator(project_dir=tmp_path)
        loc = evaluator.count_lines_of_code(test_file)

        assert 'code' in loc
        assert 'comments' in loc
        assert loc['code'] > 0


class TestIntegration:
    """Integration tests for complete pipeline."""

    @pytest.fixture
    def sample_dataset(self, tmp_path):
        """Create a minimal dataset for integration testing."""
        csv_content = """LineId,Level,Content
""" + "\n".join([f"{i},INFO,Log entry {i}" for i in range(1, 51)])

        csv_file = tmp_path / "integration_test.csv"
        csv_file.write_text(csv_content)
        return str(csv_file)

    def test_end_to_end_snn(self, sample_dataset):
        """Test complete SNN pipeline end-to-end."""
        # 1. Load data
        loader = LogDataLoader(sample_dataset)
        loader.load()
        loader.auto_detect_schema()

        sequences, labels = loader.get_sequences(window_size=5, stride=2)
        normalized = loader.normalize_for_snn(sequences, max_len=20)

        # 2. Encode
        encoder = SpikeEncoder(strategy=EncodingStrategy.RATE, time_steps=20)
        spike_trains = encoder.encode(normalized)

        # 3. Create simple labels for testing
        if len(labels) == 0:
            labels = np.random.randint(0, 2, len(spike_trains))

        # 4. Train SNN
        split = int(len(spike_trains) * 0.8)
        train_data = torch.tensor(spike_trains[:split], dtype=torch.float32)
        train_labels = torch.tensor(labels[:split], dtype=torch.long)

        model = OptimizedSpikingAnomalyDetector(
            input_size=20,
            hidden_size=32,
            output_size=2,
            time_steps=20
        )

        trainer = SNNTrainer(model, track_emissions=False, neuromorphic_speedup=1.0)

        train_loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(train_data, train_labels),
            batch_size=8
        )

        # Train for 1 epoch
        metrics = trainer.train_epoch(train_loader)

        assert 'loss' in metrics
        assert 'accuracy' in metrics
        assert metrics['accuracy'] >= 0


class TestDockerCompatibility:
    """Tests for Docker environment compatibility."""

    def test_temp_directory_access(self, tmp_path):
        """Test temporary directory creation (Docker volumes)."""
        test_file = tmp_path / "docker_test.txt"
        test_file.write_text("Docker compatibility test")

        assert test_file.exists()
        assert test_file.read_text() == "Docker compatibility test"

    def test_json_serialization(self):
        """Test JSON serialization (for Docker output)."""
        data = {
            'model': 'SNN',
            'accuracy': np.float64(95.5),
            'params': np.int64(11076)
        }

        converted = convert_to_native_types(data)
        json_str = json.dumps(converted)

        assert isinstance(json_str, str)
        parsed = json.loads(json_str)
        assert parsed['accuracy'] == 95.5


# Pytest configuration
@pytest.fixture(scope="session")
def test_config():
    """Session-wide test configuration."""
    return {
        'batch_size': 8,
        'num_epochs': 1,
        'learning_rate': 1e-3
    }


if __name__ == "__main__":
    # Run tests with: python -m pytest test_pipeline.py -v
    pytest.main([__file__, '-v', '--tb=short'])
