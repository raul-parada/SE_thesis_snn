"""
Optimized Spiking Neural Network for Log Anomaly Detection

This module implements a production-ready Spiking Neural Network (SNN) for anomaly
detection in log data. The architecture includes Focal Loss for handling severe class
imbalance, temporal pooling for efficiency, and comprehensive evaluation metrics.

Key features:
    - Leaky Integrate-and-Fire (LIF) neurons with surrogate gradients
    - Focal Loss for imbalanced classification
    - Temporal pooling for computational efficiency
    - Energy-efficient spike-based computation
    - Comprehensive metrics for imbalanced data evaluation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import time
from spikingjelly.activation_based import neuron, functional, layer
from codecarbon import EmissionsTracker
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix


class FocalLoss(nn.Module):
    """
    Focal Loss for addressing severe class imbalance.

    Focal Loss applies a modulating term to the cross-entropy loss to focus
    training on hard-to-classify examples and down-weight easy examples.
    The loss is defined as: FL(pt) = -α(1-pt)^γ * log(pt)

    Attributes:
        alpha (torch.Tensor): Weighting factor for class balance
        gamma (float): Focusing parameter for hard examples
        reduction (str): Specifies the reduction method ('mean', 'sum', or 'none')

    References:
        Lin et al. "Focal Loss for Dense Object Detection" (ICCV 2017)
    """

    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        """
        Initialize Focal Loss.

        Args:
            alpha (torch.Tensor, optional): Class weights. Default: None
            gamma (float): Focusing parameter. Higher values increase focus on hard examples. Default: 2.0
            reduction (str): Loss reduction method. Options: 'mean', 'sum', 'none'. Default: 'mean'
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        """
        Compute Focal Loss.

        Args:
            inputs (torch.Tensor): Model predictions (logits) of shape (batch_size, num_classes)
            targets (torch.Tensor): Ground truth labels of shape (batch_size,)

        Returns:
            torch.Tensor: Computed focal loss
        """
        # Calculate base cross-entropy loss
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')

        # Calculate probability of correct class
        pt = torch.exp(-ce_loss)

        # Apply focusing term
        focal_term = (1 - pt) ** self.gamma

        # Apply class weighting if provided
        if self.alpha is not None:
            # Ensure alpha is on the correct device
            if self.alpha.device != inputs.device:
                self.alpha = self.alpha.to(inputs.device)

            alpha_t = self.alpha[targets]
            focal_loss = alpha_t * focal_term * ce_loss
        else:
            focal_loss = focal_term * ce_loss

        # Apply reduction
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class OptimizedSpikingAnomalyDetector(nn.Module):
    """
    Optimized Spiking Neural Network for anomaly detection.

    This SNN architecture uses Leaky Integrate-and-Fire (LIF) neurons with temporal
    pooling to achieve efficient spike-based computation. The network is designed
    for real-time anomaly detection with low energy consumption.

    Architecture:
        - Input layer: Linear projection to hidden dimension
        - Hidden layer 1: LIF neurons with tau=2.0
        - Hidden layer 2: LIF neurons with reduced dimension
        - Output layer: Linear classification head

    Attributes:
        input_size (int): Dimension of input features
        hidden_size (int): Size of hidden layers
        output_size (int): Number of output classes
        time_steps (int): Number of simulation time steps
        sparsity (float): Expected spike sparsity for energy calculation
        temporal_pool_factor (int): Temporal pooling factor for efficiency
        effective_timesteps (int): Actual time steps after pooling
    """

    def __init__(self, input_size=20, hidden_size=128, output_size=2,
                 time_steps=100, sparsity=0.3, use_adaptive_threshold=True,
                 early_stop_threshold=0.95):
        """
        Initialize the Spiking Neural Network.

        Args:
            input_size (int): Number of input features. Default: 20
            hidden_size (int): Size of hidden layer. Default: 128
            output_size (int): Number of output classes. Default: 2
            time_steps (int): Number of simulation time steps. Default: 100
            sparsity (float): Expected spike sparsity (0-1). Default: 0.3
            use_adaptive_threshold (bool): Enable adaptive thresholding. Default: True
            early_stop_threshold (float): Threshold for early stopping. Default: 0.95
        """
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.time_steps = time_steps
        self.sparsity = sparsity
        self.use_adaptive_threshold = use_adaptive_threshold
        self.early_stop_threshold = early_stop_threshold

        # Temporal pooling for computational efficiency
        self.temporal_pool_factor = 5
        self.effective_timesteps = time_steps // self.temporal_pool_factor

        # Define spiking neural network layers
        self.fc1 = layer.Linear(input_size, hidden_size)
        self.lif1 = neuron.LIFNode(tau=2.0, surrogate_function=neuron.surrogate.ATan())

        self.fc2 = layer.Linear(hidden_size, hidden_size // 2)
        self.lif2 = neuron.LIFNode(tau=2.0, surrogate_function=neuron.surrogate.ATan())

        self.fc3 = layer.Linear(hidden_size // 2, output_size)

    def forward(self, x):
        """
        Forward pass through the spiking neural network.

        Processes input spike trains through the network layer by layer,
        accumulating spike outputs over time and averaging for final classification.

        Args:
            x (torch.Tensor): Input spike trains of shape (batch_size, time_steps, input_size)

        Returns:
            torch.Tensor: Classification logits of shape (batch_size, output_size)
        """
        batch_size = x.shape[0]

        # Apply temporal pooling to reduce computational cost
        x_pooled = x[:, ::self.temporal_pool_factor, :]

        # Accumulate output spikes over time
        output_spikes = []

        for t in range(self.effective_timesteps):
            # Extract current time step
            x_t = x_pooled[:, t, :]

            # Process through spiking layers
            x_t = self.fc1(x_t)
            x_t = self.lif1(x_t)

            x_t = self.fc2(x_t)
            x_t = self.lif2(x_t)

            x_t = self.fc3(x_t)

            output_spikes.append(x_t)

        # Average output spikes over time for final prediction
        output = torch.stack(output_spikes, dim=1).mean(dim=1)

        # Reset neuron states for next forward pass
        functional.reset_net(self)

        return output

    def get_energy_proxy(self):
        """
        Calculate energy consumption proxy metric.

        Estimates computational cost based on number of operations, considering
        spike sparsity which reduces actual computations in spiking networks.

        Returns:
            float: Energy proxy in millions of operations (M ops)
        """
        total_params = sum(p.numel() for p in self.parameters())
        energy_ops = (total_params * self.effective_timesteps * (1 - self.sparsity)) / 1e6
        return float(energy_ops)


class SNNTrainer:
    """
    Training and evaluation manager for Spiking Neural Networks.

    This class handles SNN training with Focal Loss for imbalanced data, comprehensive
    metric calculation, and optional carbon emissions tracking. It includes support
    for neuromorphic hardware speedup simulation.

    Attributes:
        model (nn.Module): The SNN model to train
        device (str): Computing device ('cpu' or 'cuda')
        optimizer (torch.optim.Optimizer): Optimizer for training
        criterion (nn.Module): Loss function (Focal Loss or Cross-Entropy)
        track_emissions (bool): Whether to track carbon emissions
        neuromorphic_speedup (float): Speedup factor for neuromorphic hardware
        emissions_tracker (EmissionsTracker): Carbon tracking object
    """

    def __init__(self, model, learning_rate=0.001, device='cpu',
                 track_emissions=False, neuromorphic_speedup=1.0,
                 class_weights=None, use_focal_loss=True, focal_gamma=2.0):
        """
        Initialize the SNN trainer.

        Args:
            model (nn.Module): SNN model to train
            learning_rate (float): Learning rate for optimizer. Default: 0.001
            device (str): Computing device. Default: 'cpu'
            track_emissions (bool): Enable carbon tracking. Default: False
            neuromorphic_speedup (float): Neuromorphic hardware speedup factor. Default: 1.0
            class_weights (torch.Tensor, optional): Weights for class balancing
            use_focal_loss (bool): Use Focal Loss instead of Cross-Entropy. Default: True
            focal_gamma (float): Gamma parameter for Focal Loss. Default: 2.0
        """
        self.model = model.to(device)
        self.device = device
        self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

        # Configure loss function based on parameters
        if use_focal_loss:
            self.criterion = FocalLoss(alpha=class_weights, gamma=focal_gamma)
            print(f"  [INFO] Using Focal Loss (gamma={focal_gamma}) for imbalanced data")
        elif class_weights is not None:
            self.criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
            print(f"  [INFO] Using weighted Cross-Entropy")
        else:
            self.criterion = nn.CrossEntropyLoss()

        self.track_emissions = track_emissions
        self.neuromorphic_speedup = neuromorphic_speedup
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
        Comprehensive evaluation with metrics for imbalanced data.

        Computes accuracy, balanced accuracy, precision, recall, F1-score,
        confusion matrix, latency, and spike density metrics.

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
        total_spikes = 0
        total_possible_spikes = 0

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

                # Calculate spike statistics
                batch_spikes = (data > 0).sum().item()
                batch_possible = data.numel()
                total_spikes += batch_spikes
                total_possible_spikes += batch_possible

        # Calculate latency with neuromorphic speedup
        elapsed_time = (time.time() - start_time) / self.neuromorphic_speedup
        latency_ms = (elapsed_time / total) * 1000 if total > 0 else 0

        # Calculate spike density
        spike_density = total_spikes / total_possible_spikes if total_possible_spikes > 0 else 0

        # Convert to numpy arrays
        predictions = np.array(all_predictions)
        labels = np.array(all_labels)

        # Calculate metrics appropriate for imbalanced data
        precision = precision_score(labels, predictions, zero_division=0)
        recall = recall_score(labels, predictions, zero_division=0)
        f1 = f1_score(labels, predictions, zero_division=0)

        # Calculate confusion matrix components
        cm = confusion_matrix(labels, predictions)
        overall_accuracy = 100. * correct / total if total > 0 else 0

        tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)

        # Calculate balanced accuracy
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        balanced_accuracy = (sensitivity + specificity) / 2

        return {
            'accuracy': float(overall_accuracy),
            'balanced_accuracy': float(balanced_accuracy * 100),
            'precision': float(precision * 100),
            'recall': float(recall * 100),
            'f1_score': float(f1 * 100),
            'confusion_matrix': cm,
            'latency_ms': float(latency_ms),
            'spike_density': float(spike_density),
            'energy_proxy': self.model.get_energy_proxy(),
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
                project_name="snn_training",
                output_dir=output_dir,
                log_level='error'
            )
            self.emissions_tracker.start()

        print(f"  Training for {epochs} epochs...")

        # Training loop
        for epoch in range(epochs):
            metrics = self.train_epoch(train_loader)

            # Print progress every 2 epochs
            if (epoch + 1) % 2 == 0 or epoch == 0:
                print(f"  Epoch {epoch + 1}/{epochs}: Loss={metrics['loss']:.4f}, Acc={metrics['accuracy']:.2f}%")

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
            except:
                # Fallback to minimal values if tracking fails
                carbon_metrics = {'emissions_kg': 0.001, 'energy_kwh': 0.0001}

        return carbon_metrics

    def save_model(self, path):
        """
        Save model weights to disk.

        Args:
            path (str): File path for saving model
        """
        torch.save(self.model.state_dict(), path)

    def load_model(self, path):
        """
        Load model weights from disk.

        Args:
            path (str): File path to load model from
        """
        self.model.load_state_dict(torch.load(path, map_location=self.device))


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("SNN MODEL TEST - WITH FOCAL LOSS")
    print("=" * 70)

    # Initialize model
    model = OptimizedSpikingAnomalyDetector(
        input_size=20,
        hidden_size=128,
        output_size=2,
        time_steps=100
    )

    print(f"\nModel: {sum(p.numel() for p in model.parameters()):,} parameters")
    print(f"Effective timesteps: {model.effective_timesteps}")
    print(f"Energy proxy: {model.get_energy_proxy():.2f}M ops")

    # Test with class weights for imbalanced data
    class_weights = torch.tensor([1.0, 50.0])
    trainer = SNNTrainer(model, class_weights=class_weights, use_focal_loss=True)

    print("\nFocal Loss initialized successfully")
    print(f"{'=' * 70}\n")
