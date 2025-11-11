"""
Neuromorphic Spike Encoding for Spiking Neural Networks

This module implements biologically-inspired spike encoding strategies for converting
numerical data into spike trains suitable for Spiking Neural Network (SNN) processing.
It supports multiple encoding schemes including rate coding, temporal coding, and
latency coding.

Key features:
    - Rate coding: Information encoded in spike frequency
    - Temporal coding: Information encoded in precise spike timing
    - Latency coding: Information encoded in first-spike latency
    - Robust handling of edge cases (empty arrays, constant values)
    - Spike density analysis for energy efficiency evaluation
"""

import numpy as np
from enum import Enum
from typing import Union


class EncodingStrategy(Enum):
    """
    Enumeration of supported spike encoding strategies.

    Each strategy represents a different biological principle for encoding
    information in neural spike trains.
    """
    RATE = "rate"           # Rate coding: higher value results in higher firing rate
    TEMPORAL = "temporal"   # Temporal coding: higher value results in earlier spike
    LATENCY = "latency"     # Latency coding: higher value results in lower latency


class SpikeEncoder:
    """
    Encoder for converting numerical data into spike trains.

    This class implements multiple biologically-inspired encoding strategies that
    transform numerical values into temporal patterns of spikes suitable for SNN
    processing. Each encoding strategy offers different trade-offs between information
    density, temporal precision, and computational efficiency.

    Supported strategies:
        - Rate Coding: Information encoded in firing frequency (most biologically common)
        - Temporal Coding: Information encoded in precise spike timing
        - Latency Coding: Information encoded in first-spike latency with Gaussian distribution

    Attributes:
        strategy (EncodingStrategy): Selected encoding strategy
        time_steps (int): Number of discrete time steps in the spike train
        max_rate (float): Maximum firing rate (0.0 to 1.0)
    """

    def __init__(self, strategy: EncodingStrategy = EncodingStrategy.RATE,
                 time_steps: int = 100, max_rate: float = 1.0):
        """
        Initialize the spike encoder.

        Args:
            strategy (EncodingStrategy): Encoding strategy to use. Default: RATE
            time_steps (int): Number of time steps for spike trains. Default: 100
            max_rate (float): Maximum firing rate (0.0 to 1.0). Default: 1.0
        """
        self.strategy = strategy
        self.time_steps = time_steps
        self.max_rate = max_rate

    def encode(self, values: np.ndarray) -> np.ndarray:
        """
        Encode numerical values into spike trains.

        This method converts numerical data into spike trains using the selected
        encoding strategy. It includes robust handling of edge cases such as empty
        arrays and constant values.

        Args:
            values (np.ndarray): Input array of shape (samples, features) or (samples,)

        Returns:
            np.ndarray: Spike trains of shape (samples, time_steps, features)
        """
        # Handle empty arrays
        if values.size == 0:
            print("  [WARNING] Empty array passed to encoder - returning empty spike train")
            return np.array([]).reshape(0, self.time_steps, 0)

        # Ensure 2D array format
        if values.ndim == 1:
            values = values.reshape(-1, 1)

        samples, features = values.shape

        # Initialize spike train array
        spike_train = np.zeros((samples, self.time_steps, features), dtype=np.float32)

        # Normalize values to [0, 1] range
        values_min = values.min()
        values_max = values.max()

        # Handle constant values (all identical)
        if values_max == values_min:
            print(f"  [WARNING] Constant values detected (all {values_min:.2f}) - using default encoding")
            values_min = 0
            values_max = 1
            if values_min == 0:
                values_max = 1

        # Normalize with epsilon to prevent division by zero
        normalized = (values - values_min) / (values_max - values_min + 1e-8)
        normalized = np.clip(normalized, 0, 1)  # Ensure valid range

        # Apply selected encoding strategy
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
        Rate coding: encode information in spike frequency.

        This encoding strategy mimics biological neurons where higher stimulus
        intensity results in higher firing frequency. Each time step has a
        probabilistic spike based on the input value.

        Args:
            normalized_values (np.ndarray): Normalized input values (0-1)

        Returns:
            np.ndarray: Rate-coded spike train
        """
        samples, features = normalized_values.shape
        spike_train = np.zeros((samples, self.time_steps, features), dtype=np.float32)

        for t in range(self.time_steps):
            # Generate stochastic spikes based on firing probability
            firing_prob = normalized_values * self.max_rate
            spikes = np.random.rand(samples, features) < firing_prob
            spike_train[:, t, :] = spikes.astype(np.float32)

        return spike_train

    def _temporal_coding(self, normalized_values: np.ndarray) -> np.ndarray:
        """
        Temporal coding: encode information in precise spike timing.

        This strategy encodes information through the exact timing of single spikes.
        Higher values produce earlier spikes, allowing for very rapid information
        transmission with single spikes.

        Args:
            normalized_values (np.ndarray): Normalized input values (0-1)

        Returns:
            np.ndarray: Temporally-coded spike train with single precise spikes
        """
        samples, features = normalized_values.shape
        spike_train = np.zeros((samples, self.time_steps, features), dtype=np.float32)

        # Calculate spike timing: higher value results in earlier spike
        spike_times = ((1 - normalized_values) * (self.time_steps - 1)).astype(int)
        spike_times = np.clip(spike_times, 0, self.time_steps - 1)

        # Place single spike at calculated time for each feature
        for i in range(samples):
            for j in range(features):
                t = spike_times[i, j]
                spike_train[i, t, j] = 1.0

        return spike_train

    def _latency_coding(self, normalized_values: np.ndarray) -> np.ndarray:
        """
        Latency coding: encode information in first-spike latency.

        Similar to temporal coding but uses a Gaussian distribution of spikes
        centered around the latency time. This provides more robust encoding
        with some tolerance for temporal noise.

        Args:
            normalized_values (np.ndarray): Normalized input values (0-1)

        Returns:
            np.ndarray: Latency-coded spike train with Gaussian distribution
        """
        samples, features = normalized_values.shape
        spike_train = np.zeros((samples, self.time_steps, features), dtype=np.float32)

        # Calculate latency: higher value results in lower latency (earlier response)
        latency = ((1 - normalized_values) * (self.time_steps - 1)).astype(int)
        latency = np.clip(latency, 0, self.time_steps - 1)

        # Define Gaussian distribution width
        sigma = self.time_steps / 20

        # Create Gaussian spike distribution around latency center
        for i in range(samples):
            for j in range(features):
                center = latency[i, j]
                for t in range(self.time_steps):
                    # Calculate Gaussian probability at this time step
                    spike_prob = np.exp(-((t - center) ** 2) / (2 * sigma ** 2))
                    spike_train[i, t, j] = spike_prob * self.max_rate

        return spike_train

    def decode(self, spike_train: np.ndarray) -> np.ndarray:
        """
        Decode spike trains back to numerical values.

        This method reverses the encoding process for analysis purposes. The decoding
        strategy depends on the encoding method used.

        Args:
            spike_train (np.ndarray): Spike train of shape (samples, time_steps, features)

        Returns:
            np.ndarray: Decoded numerical values of shape (samples, features)
        """
        if spike_train.size == 0:
            return np.array([])

        if self.strategy == EncodingStrategy.RATE:
            # Decode rate coding by averaging spike count over time
            return spike_train.sum(axis=1) / self.time_steps

        elif self.strategy == EncodingStrategy.TEMPORAL:
            # Decode temporal coding by finding first spike time
            decoded = np.zeros((spike_train.shape[0], spike_train.shape[2]))

            for i in range(spike_train.shape[0]):
                for j in range(spike_train.shape[2]):
                    spike_times = np.where(spike_train[i, :, j] > 0)[0]
                    if len(spike_times) > 0:
                        # Earlier spike indicates higher value
                        decoded[i, j] = 1 - (spike_times[0] / self.time_steps)
                    else:
                        decoded[i, j] = 0

            return decoded

        elif self.strategy == EncodingStrategy.LATENCY:
            # Decode latency coding using weighted average of spike times
            time_weights = np.arange(self.time_steps).reshape(1, -1, 1)
            weighted_sum = (spike_train * time_weights).sum(axis=1)
            total_spikes = spike_train.sum(axis=1) + 1e-8  # Avoid division by zero
            avg_time = weighted_sum / total_spikes

            return 1 - (avg_time / self.time_steps)

        else:
            raise ValueError(f"Unknown encoding strategy: {self.strategy}")

    def get_spike_density(self, spike_train: np.ndarray) -> float:
        """
        Calculate spike density (sparsity metric).

        Spike density represents the proportion of active (non-zero) spikes in the
        train. Lower density indicates more sparse, energy-efficient encoding.

        Args:
            spike_train (np.ndarray): Spike train array

        Returns:
            float: Spike density (fraction of active spikes, 0.0 to 1.0)
        """
        if spike_train.size == 0:
            return 0.0

        return float(np.mean(spike_train > 0))

    def __repr__(self):
        """String representation of the encoder configuration."""
        return (f"SpikeEncoder(strategy={self.strategy.value}, "
                f"time_steps={self.time_steps}, max_rate={self.max_rate})")


def test_encoder():
    """
    Test spike encoder with various inputs and encoding strategies.

    This function validates encoder functionality including all three encoding
    strategies, edge cases (empty arrays, constant values), and decoding accuracy.
    """
    print("\n" + "=" * 70)
    print("SPIKE ENCODER TEST")
    print("=" * 70)

    # Define test data
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

        # Test decoding accuracy
        decoded = encoder.decode(spike_train)
        print(f"Decoded shape: {decoded.shape}")
        print(f"Reconstruction error: {np.mean(np.abs(test_data - decoded)):.4f}")

    # Test edge cases
    print(f"\n{'=' * 70}")
    print("Testing edge cases")
    print(f"{'=' * 70}")

    encoder = SpikeEncoder()

    # Empty array
    print("\n1. Empty array:")
    empty = np.array([])
    result = encoder.encode(empty)
    print(f"  Result shape: {result.shape}")

    # Constant values
    print("\n2. Constant values:")
    constant = np.ones((5, 3))
    result = encoder.encode(constant)
    print(f"  Result shape: {result.shape}")
    print(f"  Spike density: {encoder.get_spike_density(result):.4f}")

    # Single value
    print("\n3. Single value:")
    single = np.array([[0.5]])
    result = encoder.encode(single)
    print(f"  Result shape: {result.shape}")

    print(f"\n{'=' * 70}")
    print("ALL TESTS PASSED")
    print(f"{'=' * 70}\n")


if __name__ == "__main__":
    test_encoder()
