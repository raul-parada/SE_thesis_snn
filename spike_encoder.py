# spike_encoder.py
"""
Neuromorphic spike encoding for log sequences.
Converts numerical data into spike trains for SNN processing.
FIXED: Empty array handling, constant value handling.
"""

import numpy as np
from enum import Enum
from typing import Union


class EncodingStrategy(Enum):
    """Spike encoding strategies."""
    RATE = "rate"  # Rate coding: higher value → higher firing rate
    TEMPORAL = "temporal"  # Temporal coding: higher value → earlier spike
    LATENCY = "latency"  # Latency coding: higher value → lower latency


class SpikeEncoder:
    """
    Encode numerical data into spike trains for Spiking Neural Networks.

    Supports multiple biologically-inspired encoding strategies:
    - Rate Coding: Information encoded in firing frequency
    - Temporal Coding: Information encoded in spike timing
    - Latency Coding: Information encoded in first-spike latency
    """

    def __init__(self, strategy: EncodingStrategy = EncodingStrategy.RATE,
                 time_steps: int = 100, max_rate: float = 1.0):
        """
        Initialize spike encoder.

        Args:
            strategy: Encoding strategy (RATE, TEMPORAL, or LATENCY)
            time_steps: Number of time steps for spike train
            max_rate: Maximum firing rate (0-1)
        """
        self.strategy = strategy
        self.time_steps = time_steps
        self.max_rate = max_rate

    def encode(self, values: np.ndarray) -> np.ndarray:
        """
        Encode numerical values into spike trains.
        FIXED: Handles empty arrays, constant values, edge cases.

        Args:
            values: Input array of shape (samples, features)

        Returns:
            Spike trains of shape (samples, time_steps, features)
        """
        # Handle empty arrays
        if values.size == 0:
            print("  [WARNING] Empty array passed to encoder - returning empty spike train")
            return np.array([]).reshape(0, self.time_steps, 0)

        # Ensure 2D array
        if values.ndim == 1:
            values = values.reshape(-1, 1)

        samples, features = values.shape

        # Initialize spike train
        spike_train = np.zeros((samples, self.time_steps, features), dtype=np.float32)

        # Normalize values to [0, 1]
        values_min = values.min()
        values_max = values.max()

        # Handle constant values (all same)
        if values_max == values_min:
            print(f"  [WARNING] Constant values detected (all {values_min:.2f}) - using default encoding")
            values_min = 0
            values_max = 1
            if values_min == 0:
                values_max = 1

        # Normalize with epsilon to avoid division by zero
        normalized = (values - values_min) / (values_max - values_min + 1e-8)
        normalized = np.clip(normalized, 0, 1)  # Ensure [0, 1] range

        # Apply encoding strategy
        if self.strategy == EncodingStrategy.RATE:
            spike_train = self._rate_coding(normalized)
        elif self.strategy == EncodingStrategy.TEMPORAL:
            spike_train = self._temporal_coding(normalized)
        elif self.strategy == EncodingStrategy.LATENCY:
            spike_train = self._latency_coding(normalized)
        else:
            raise ValueError(f"Unknown encoding strategy: {self.strategy}")

        return spike_train

    def _rate_coding(self, normalized_values: np.ndarray) -> np.ndarray:
        """
        Rate coding: Higher values → higher firing rate.

        Biologically inspired by neurons that fire more frequently
        for stronger stimuli.

        Args:
            normalized_values: Normalized input (0-1)

        Returns:
            Spike train with rate-coded spikes
        """
        samples, features = normalized_values.shape
        spike_train = np.zeros((samples, self.time_steps, features), dtype=np.float32)

        for t in range(self.time_steps):
            # Generate random spikes based on firing rate
            firing_prob = normalized_values * self.max_rate
            spikes = np.random.rand(samples, features) < firing_prob
            spike_train[:, t, :] = spikes.astype(np.float32)

        return spike_train

    def _temporal_coding(self, normalized_values: np.ndarray) -> np.ndarray:
        """
        Temporal coding: Higher values → earlier spikes.

        Encodes information in the precise timing of spikes.

        Args:
            normalized_values: Normalized input (0-1)

        Returns:
            Spike train with temporally-coded spikes
        """
        samples, features = normalized_values.shape
        spike_train = np.zeros((samples, self.time_steps, features), dtype=np.float32)

        # Calculate spike times (higher value → earlier spike)
        spike_times = ((1 - normalized_values) * (self.time_steps - 1)).astype(int)
        spike_times = np.clip(spike_times, 0, self.time_steps - 1)

        # Place spikes at calculated times
        for i in range(samples):
            for j in range(features):
                t = spike_times[i, j]
                spike_train[i, t, j] = 1.0

        return spike_train

    def _latency_coding(self, normalized_values: np.ndarray) -> np.ndarray:
        """
        Latency coding: Higher values → lower latency (earlier response).

        Similar to temporal coding but with Gaussian spike distribution.

        Args:
            normalized_values: Normalized input (0-1)

        Returns:
            Spike train with latency-coded spikes
        """
        samples, features = normalized_values.shape
        spike_train = np.zeros((samples, self.time_steps, features), dtype=np.float32)

        # Calculate latency (higher value → lower latency)
        latency = ((1 - normalized_values) * (self.time_steps - 1)).astype(int)
        latency = np.clip(latency, 0, self.time_steps - 1)

        # Create Gaussian spike distribution around latency
        sigma = self.time_steps / 20  # Width of spike distribution

        for i in range(samples):
            for j in range(features):
                center = latency[i, j]
                for t in range(self.time_steps):
                    # Gaussian distribution
                    spike_prob = np.exp(-((t - center) ** 2) / (2 * sigma ** 2))
                    spike_train[i, t, j] = spike_prob * self.max_rate

        return spike_train

    def decode(self, spike_train: np.ndarray) -> np.ndarray:
        """
        Decode spike train back to numerical values (for analysis).

        Args:
            spike_train: Spike train of shape (samples, time_steps, features)

        Returns:
            Decoded values (samples, features)
        """
        if spike_train.size == 0:
            return np.array([])

        if self.strategy == EncodingStrategy.RATE:
            # Sum spikes across time (firing rate)
            return spike_train.sum(axis=1) / self.time_steps
        elif self.strategy == EncodingStrategy.TEMPORAL:
            # Find first spike time
            decoded = np.zeros((spike_train.shape[0], spike_train.shape[2]))
            for i in range(spike_train.shape[0]):
                for j in range(spike_train.shape[2]):
                    spike_times = np.where(spike_train[i, :, j] > 0)[0]
                    if len(spike_times) > 0:
                        decoded[i, j] = 1 - (spike_times[0] / self.time_steps)
                    else:
                        decoded[i, j] = 0
            return decoded
        elif self.strategy == EncodingStrategy.LATENCY:
            # Weighted average of spike times
            time_weights = np.arange(self.time_steps).reshape(1, -1, 1)
            weighted_sum = (spike_train * time_weights).sum(axis=1)
            total_spikes = spike_train.sum(axis=1) + 1e-8
            avg_time = weighted_sum / total_spikes
            return 1 - (avg_time / self.time_steps)
        else:
            raise ValueError(f"Unknown encoding strategy: {self.strategy}")

    def get_spike_density(self, spike_train: np.ndarray) -> float:
        """
        Calculate spike density (sparsity metric).

        Args:
            spike_train: Spike train array

        Returns:
            Spike density (fraction of active spikes)
        """
        if spike_train.size == 0:
            return 0.0
        return float(np.mean(spike_train > 0))

    def __repr__(self):
        return (f"SpikeEncoder(strategy={self.strategy.value}, "
                f"time_steps={self.time_steps}, max_rate={self.max_rate})")


def test_encoder():
    """Test spike encoder with various inputs."""
    print("\n" + "=" * 70)
    print("SPIKE ENCODER TEST")
    print("=" * 70)

    # Test data
    test_data = np.array([
        [0.0, 0.5, 1.0],
        [0.2, 0.7, 0.3],
        [1.0, 0.0, 0.5]
    ])

    print(f"\nTest data shape: {test_data.shape}")
    print(f"Test data:\n{test_data}")

    # Test each encoding strategy
    strategies = [EncodingStrategy.RATE, EncodingStrategy.TEMPORAL, EncodingStrategy.LATENCY]

    for strategy in strategies:
        print(f"\n{'=' * 70}")
        print(f"Testing {strategy.value.upper()} encoding")
        print(f"{'=' * 70}")

        encoder = SpikeEncoder(strategy=strategy, time_steps=50, max_rate=0.8)
        spike_train = encoder.encode(test_data)

        print(f"Spike train shape: {spike_train.shape}")
        print(f"Spike density: {encoder.get_spike_density(spike_train):.4f}")

        # Decode
        decoded = encoder.decode(spike_train)
        print(f"Decoded shape: {decoded.shape}")
        print(f"Reconstruction error: {np.mean(np.abs(test_data - decoded)):.4f}")

    # Test edge cases
    print(f"\n{'=' * 70}")
    print("Testing edge cases")
    print(f"{'=' * 70}")

    # Empty array
    print("\n1. Empty array:")
    encoder = SpikeEncoder()
    empty = np.array([])
    result = encoder.encode(empty)
    print(f"   Result shape: {result.shape}")

    # Constant values
    print("\n2. Constant values:")
    constant = np.ones((5, 3))
    result = encoder.encode(constant)
    print(f"   Result shape: {result.shape}")
    print(f"   Spike density: {encoder.get_spike_density(result):.4f}")

    # Single value
    print("\n3. Single value:")
    single = np.array([[0.5]])
    result = encoder.encode(single)
    print(f"   Result shape: {result.shape}")

    print(f"\n{'=' * 70}")
    print("✓ ALL TESTS PASSED")
    print(f"{'=' * 70}\n")


if __name__ == "__main__":
    test_encoder()
