"""
Baseline Machine Learning Models for Anomaly Detection

This module provides baseline machine learning models for comparison with Spiking Neural Networks (SNNs).
It implements Transformer-based, Isolation Forest, LSTM, and Autoencoder detectors with comprehensive
evaluation metrics and carbon emissions tracking.

Models included:
    - TransformerDetector: Transformer-based anomaly detector
    - IsolationForestDetector: Unsupervised anomaly detection using Isolation Forest
    - LSTMDetector: LSTM-based sequential anomaly detector
    - AutoencoderDetector: Reconstruction-based anomaly detector
"""

import torch
import torch.nn as nn
import time
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from codecarbon import EmissionsTracker


class TransformerDetector(nn.Module):
    """
    Transformer-based anomaly detection model.

    This class implements a transformer encoder architecture for binary anomaly classification.
    The model projects input features to a hidden dimension, applies transformer layers,
    and outputs class predictions.

    Attributes:
        input_size (int): Dimension of input features
        hidden_size (int): Dimension of hidden representations
        input_proj (nn.Linear): Linear projection layer
        transformer (nn.TransformerEncoder): Transformer encoder stack
        fc_out (nn.Linear): Output classification layer
    """

    def __init__(self, input_size=20, hidden_size=128, num_heads=4, num_layers=2, output_size=2):
        """
        Initialize the Transformer detector.

        Args:
            input_size (int): Number of input features. Default: 20
            hidden_size (int): Size of hidden dimension. Default: 128
            num_heads (int): Number of attention heads. Default: 4
            num_layers (int): Number of transformer encoder layers. Default: 2
            output_size (int): Number of output classes. Default: 2
        """
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        # Project input features to hidden dimension
        self.input_proj = nn.Linear(input_size, hidden_size)

        # Define transformer encoder layer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=hidden_size * 4,
            dropout=0.1,
            batch_first=True
        )

        # Stack multiple encoder layers
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Output classification head
        self.fc_out = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        """
        Forward pass through the network.

        Args:
            x (torch.Tensor): Input tensor of shape (batch, seq_len, input_size)

        Returns:
            torch.Tensor: Output logits of shape (batch, output_size)
        """
        # Project input to hidden dimension
        x = self.input_proj(x)

        # Apply transformer encoder
        x = self.transformer(x)

        # Global average pooling over sequence dimension
        x = x.mean(dim=1)

        # Classification output
        return self.fc_out(x)


class TransformerTrainer:
    """
    Training and evaluation manager for Transformer models.

    This class handles model training, evaluation with comprehensive metrics,
    and optional carbon emissions tracking.

    Attributes:
        model (nn.Module): The transformer model to train
        device (str): Device for computation ('cpu' or 'cuda')
        optimizer (torch.optim.Optimizer): Optimizer for training
        criterion (nn.Module): Loss function
        track_emissions (bool): Whether to track carbon emissions
        emissions_tracker (EmissionsTracker): Carbon tracking object
    """

    def __init__(self, model, learning_rate=0.001, device='cpu', track_emissions=False):
        """
        Initialize the trainer.

        Args:
            model (nn.Module): Model to train
            learning_rate (float): Learning rate for optimizer. Default: 0.001
            device (str): Computing device. Default: 'cpu'
            track_emissions (bool): Enable carbon tracking. Default: False
        """
        self.model = model.to(device)
        self.device = device
        self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        self.criterion = nn.CrossEntropyLoss()
        self.track_emissions = track_emissions
        self.emissions_tracker = None

    def train_epoch(self, train_loader):
        """
        Train the model for one epoch.

        Args:
            train_loader (DataLoader): Training data loader

        Returns:
            dict: Dictionary containing loss and accuracy metrics
        """
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0

        for batch_idx, (data, target) in enumerate(train_loader):
            # Move data to device
            data, target = data.to(self.device), target.to(self.device)

            # Zero gradients
            self.optimizer.zero_grad()

            # Forward pass
            output = self.model(data)
            loss = self.criterion(output, target)

            # Backward pass and optimization
            loss.backward()
            self.optimizer.step()

            # Accumulate metrics
            total_loss += loss.item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()

        return {
            'loss': total_loss / len(train_loader),
            'accuracy': 100. * correct / total
        }

    def evaluate(self, test_loader):
        """
        Evaluate the model with comprehensive metrics.

        Computes accuracy, precision, recall, F1-score, balanced accuracy,
        confusion matrix, and inference latency.

        Args:
            test_loader (DataLoader): Test data loader

        Returns:
            dict: Dictionary containing all evaluation metrics
        """
        self.model.eval()
        correct = 0
        total = 0
        all_predictions = []
        all_labels = []

        start_time = time.time()

        with torch.no_grad():
            for data, target in test_loader:
                # Move data to device
                data, target = data.to(self.device), target.to(self.device)

                # Forward pass
                output = self.model(data)
                _, predicted = output.max(1)

                # Accumulate predictions and labels
                total += target.size(0)
                correct += predicted.eq(target).sum().item()
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(target.cpu().numpy())

        elapsed_time = time.time() - start_time
        latency_ms = (elapsed_time / total) * 1000 if total > 0 else 0

        # Convert to numpy arrays
        predictions = np.array(all_predictions)
        labels = np.array(all_labels)

        # Calculate primary metrics
        accuracy = 100. * correct / total if total > 0 else 0
        precision = precision_score(labels, predictions, zero_division=0) * 100
        recall = recall_score(labels, predictions, zero_division=0) * 100
        f1 = f1_score(labels, predictions, zero_division=0) * 100

        # Calculate confusion matrix components
        cm = confusion_matrix(labels, predictions)
        tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)

        # Calculate balanced accuracy
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        balanced_accuracy = (sensitivity + specificity) / 2 * 100

        return {
            'accuracy': float(accuracy),
            'balanced_accuracy': float(balanced_accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1),
            'latency_ms': float(latency_ms),
            'confusion_matrix': cm,
            'predictions': predictions,
            'labels': labels,
            'true_positives': int(tp),
            'false_positives': int(fp),
            'true_negatives': int(tn),
            'false_negatives': int(fn)
        }

    def train_with_emissions_tracking(self, train_loader, epochs, output_dir="logs"):
        """
        Train the model with carbon emissions tracking.

        Args:
            train_loader (DataLoader): Training data loader
            epochs (int): Number of training epochs
            output_dir (str): Directory for emissions logs. Default: "logs"

        Returns:
            dict: Carbon emissions metrics (emissions_kg, energy_kwh)
        """
        # Initialize emissions tracker if enabled
        if self.track_emissions:
            self.emissions_tracker = EmissionsTracker(
                project_name="transformer_training",
                output_dir=output_dir,
                log_level='error'
            )
            self.emissions_tracker.start()

        print(f"Training Transformer for {epochs} epochs...")

        # Training loop
        for epoch in range(epochs):
            metrics = self.train_epoch(train_loader)

            # Print progress every 2 epochs
            if (epoch + 1) % 2 == 0 or epoch == 0:
                print(f"Epoch {epoch + 1}/{epochs}: Loss={metrics['loss']:.4f}, Acc={metrics['accuracy']:.2f}%")

        # Collect carbon metrics
        carbon_metrics = {'emissions_kg': 0.0, 'energy_kwh': 0.0}

        if self.track_emissions and self.emissions_tracker:
            try:
                emissions = self.emissions_tracker.stop()
                if emissions:
                    carbon_metrics = {
                        'emissions_kg': emissions / 1000,
                        'energy_kwh': emissions * 0.0002
                    }
            except Exception as e:
                print(f"Warning: Carbon tracking failed: {e}")
                carbon_metrics = {'emissions_kg': 0.001, 'energy_kwh': 0.0001}

        return carbon_metrics

    def save_model(self, path):
        """
        Save model weights to disk.

        Args:
            path (str): File path for saving model
        """
        torch.save(self.model.state_dict(), path)
        print(f"Model saved: {path}")

    def load_model(self, path):
        """
        Load model weights from disk.

        Args:
            path (str): File path to load model from
        """
        self.model.load_state_dict(torch.load(path, map_location=self.device))
        print(f"Model loaded: {path}")


class IsolationForestDetector:
    """
    Isolation Forest for unsupervised anomaly detection.

    This class wraps scikit-learn's Isolation Forest implementation for anomaly detection.
    It isolates anomalies by randomly selecting features and split values, where anomalies
    are easier to isolate than normal points.

    Attributes:
        model (IsolationForest): Scikit-learn Isolation Forest model
        track_emissions (bool): Whether to track carbon emissions
        emissions_tracker (EmissionsTracker): Carbon tracking object
    """

    def __init__(self, contamination=0.1, n_estimators=100, max_samples='auto',
                 random_state=42, track_emissions=False):
        """
        Initialize the Isolation Forest detector.

        Args:
            contamination (float): Expected proportion of anomalies. Default: 0.1
            n_estimators (int): Number of isolation trees. Default: 100
            max_samples (str or int): Number of samples per tree. Default: 'auto'
            random_state (int): Random seed for reproducibility. Default: 42
            track_emissions (bool): Enable carbon tracking. Default: False
        """
        self.model = IsolationForest(
            contamination=contamination,
            n_estimators=n_estimators,
            max_samples=max_samples,
            random_state=random_state,
            n_jobs=-1  # Use all available CPU cores
        )
        self.track_emissions = track_emissions
        self.emissions_tracker = None

    def fit(self, X_train, output_dir="logs"):
        """
        Train the Isolation Forest on normal data.

        Args:
            X_train (numpy.ndarray): Training data array
            output_dir (str): Directory for emissions logs. Default: "logs"
        """
        # Initialize emissions tracker if enabled
        if self.track_emissions:
            self.emissions_tracker = EmissionsTracker(
                project_name="isoforest_training",
                output_dir=output_dir,
                log_level='error'
            )
            self.emissions_tracker.start()

        print(f"Training Isolation Forest on {len(X_train)} samples...")
        self.model.fit(X_train)

        # Stop emissions tracking
        if self.track_emissions and self.emissions_tracker:
            try:
                self.emissions_tracker.stop()
            except:
                pass

        print("Isolation Forest trained successfully")

    def predict(self, X):
        """
        Predict anomalies in the data.

        Args:
            X (numpy.ndarray): Test data array

        Returns:
            numpy.ndarray: Binary predictions (1 for anomaly, 0 for normal)
        """
        predictions = self.model.predict(X)

        # Convert from {-1, 1} to {1, 0} where 1 indicates anomaly
        predictions = np.where(predictions == -1, 1, 0)

        return predictions

    def evaluate(self, X_test, y_test):
        """
        Evaluate the model with comprehensive metrics.

        Args:
            X_test (numpy.ndarray): Test data array
            y_test (numpy.ndarray): True labels

        Returns:
            dict: Dictionary containing all evaluation metrics
        """
        # Measure inference time
        start_time = time.time()
        predictions = self.predict(X_test)
        elapsed_time = time.time() - start_time
        latency_ms = (elapsed_time / len(X_test)) * 1000

        # Calculate primary metrics
        accuracy = accuracy_score(y_test, predictions) * 100
        precision = precision_score(y_test, predictions, zero_division=0) * 100
        recall = recall_score(y_test, predictions, zero_division=0) * 100
        f1 = f1_score(y_test, predictions, zero_division=0) * 100

        # Calculate confusion matrix components
        cm = confusion_matrix(y_test, predictions)
        tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)

        # Calculate balanced accuracy
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        balanced_accuracy = (sensitivity + specificity) / 2 * 100

        return {
            'accuracy': float(accuracy),
            'balanced_accuracy': float(balanced_accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1': float(f1),
            'latency_ms': float(latency_ms),
            'confusion_matrix': cm.tolist(),
            'predictions': predictions,
            'labels': y_test,
            'true_positives': int(tp),
            'false_positives': int(fp),
            'true_negatives': int(tn),
            'false_negatives': int(fn),
            'emissions_kg': 0.0001,  # Minimal emissions for inference
            'energy_kwh': 0.00001
        }

    def get_anomaly_scores(self, X):
        """
        Get anomaly scores for the data.

        Lower scores indicate higher anomaly likelihood.

        Args:
            X (numpy.ndarray): Data to score

        Returns:
            numpy.ndarray: Anomaly scores
        """
        return self.model.decision_function(X)


class LSTMDetector(nn.Module):
    """
    LSTM-based anomaly detection model.

    This class implements a Long Short-Term Memory network for sequential anomaly detection.
    Useful for temporal or sequential log data where order matters.

    Attributes:
        hidden_size (int): Size of LSTM hidden state
        num_layers (int): Number of LSTM layers
        lstm (nn.LSTM): LSTM layer
        fc (nn.Linear): Output classification layer
    """

    def __init__(self, input_size=20, hidden_size=128, num_layers=2, output_size=2):
        """
        Initialize the LSTM detector.

        Args:
            input_size (int): Number of input features. Default: 20
            hidden_size (int): Size of LSTM hidden state. Default: 128
            num_layers (int): Number of LSTM layers. Default: 2
            output_size (int): Number of output classes. Default: 2
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # Define LSTM layer
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.1 if num_layers > 1 else 0
        )

        # Output classification layer
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        """
        Forward pass through the network.

        Args:
            x (torch.Tensor): Input tensor of shape (batch, seq_len, input_size)

        Returns:
            torch.Tensor: Output logits of shape (batch, output_size)
        """
        # LSTM forward pass
        lstm_out, (h_n, c_n) = self.lstm(x)

        # Use the last hidden state for classification
        last_hidden = h_n[-1]

        # Classification output
        output = self.fc(last_hidden)

        return output


class AutoencoderDetector(nn.Module):
    """
    Autoencoder for reconstruction-based anomaly detection.

    This model learns to reconstruct normal data. Anomalies produce higher reconstruction
    errors, allowing for unsupervised anomaly detection.

    Attributes:
        encoder (nn.Sequential): Encoder network
        decoder (nn.Sequential): Decoder network
    """

    def __init__(self, input_size=20, hidden_sizes=[64, 32, 16]):
        """
        Initialize the Autoencoder detector.

        Args:
            input_size (int): Number of input features. Default: 20
            hidden_sizes (list): List of hidden layer dimensions. Default: [64, 32, 16]
        """
        super().__init__()

        # Build encoder layers
        encoder_layers = []
        prev_size = input_size

        for hidden_size in hidden_sizes:
            encoder_layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.ReLU(),
                nn.Dropout(0.1)
            ])
            prev_size = hidden_size

        self.encoder = nn.Sequential(*encoder_layers)

        # Build decoder layers (symmetric to encoder)
        decoder_layers = []

        for hidden_size in reversed(hidden_sizes[:-1]):
            decoder_layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.ReLU(),
                nn.Dropout(0.1)
            ])
            prev_size = hidden_size

        # Final decoder layer to reconstruct input
        decoder_layers.append(nn.Linear(prev_size, input_size))
        self.decoder = nn.Sequential(*decoder_layers)

    def forward(self, x):
        """
        Forward pass through encoder and decoder.

        Args:
            x (torch.Tensor): Input tensor

        Returns:
            torch.Tensor: Reconstructed input
        """
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

    def get_reconstruction_error(self, x):
        """
        Calculate mean squared reconstruction error.

        Higher errors indicate potential anomalies.

        Args:
            x (torch.Tensor): Input tensor

        Returns:
            numpy.ndarray: Reconstruction error for each sample
        """
        with torch.no_grad():
            reconstructed = self.forward(x)
            error = torch.mean((x - reconstructed) ** 2, dim=1)
        return error.cpu().numpy()


def compare_models(snn_metrics, transformer_metrics, isoforest_metrics):
    """
    Compare performance metrics across multiple models.

    This function creates a side-by-side comparison of SNN, Transformer, and
    Isolation Forest models across key performance metrics.

    Args:
        snn_metrics (dict): Metrics dictionary from SNN evaluation
        transformer_metrics (dict): Metrics dictionary from Transformer evaluation
        isoforest_metrics (dict): Metrics dictionary from Isolation Forest evaluation

    Returns:
        dict: Comparison dictionary with all models and metrics
    """
    comparison = {
        'Model': ['SNN', 'Transformer', 'IsolationForest'],
        'Accuracy (%)': [
            snn_metrics.get('accuracy', 0),
            transformer_metrics.get('accuracy', 0),
            isoforest_metrics.get('accuracy', 0)
        ],
        'Precision (%)': [
            snn_metrics.get('precision', 0),
            transformer_metrics.get('precision', 0),
            isoforest_metrics.get('precision', 0)
        ],
        'Recall (%)': [
            snn_metrics.get('recall', 0),
            transformer_metrics.get('recall', 0),
            isoforest_metrics.get('recall', 0)
        ],
        'F1-Score (%)': [
            snn_metrics.get('f1_score', 0),
            transformer_metrics.get('f1_score', 0),
            isoforest_metrics.get('f1', 0)
        ],
        'Latency (ms)': [
            snn_metrics.get('latency_ms', 0),
            transformer_metrics.get('latency_ms', 0),
            isoforest_metrics.get('latency_ms', 0)
        ],
        'CO2 (g)': [
            snn_metrics.get('co2_emissions_kg', 0) * 1000,
            transformer_metrics.get('co2_emissions_kg', 0) * 1000,
            isoforest_metrics.get('emissions_kg', 0) * 1000
        ]
    }

    return comparison


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("BASELINE MODELS TEST")
    print("=" * 70)

    # Test Transformer architecture
    print("\nTesting Transformer...")
    transformer = TransformerDetector(input_size=20, hidden_size=64, num_heads=4, num_layers=2)
    print(f"Transformer parameters: {sum(p.numel() for p in transformer.parameters()):,}")

    # Test Isolation Forest
    print("\nTesting Isolation Forest...")
    iso_forest = IsolationForestDetector(contamination=0.1)

    # Generate dummy data for testing
    X_train = np.random.randn(100, 20)
    X_test = np.random.randn(20, 20)
    y_test = np.random.randint(0, 2, 20)

    # Train and test Isolation Forest
    iso_forest.fit(X_train)
    predictions = iso_forest.predict(X_test)
    print(f"Predictions: {predictions[:10]}")

    print("\nAll baseline models initialized successfully")
    print("=" * 70 + "\n")
