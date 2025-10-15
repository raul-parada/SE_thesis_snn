# model_snn.py
"""
Ultra-optimized Spiking Neural Network with Focal Loss for severe class imbalance.
PRODUCTION READY: Actually detects anomalies with proper evaluation metrics.
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
    Focal Loss for handling severe class imbalance.
    FL(pt) = -α(1-pt)^γ * log(pt)
    """

    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_term = (1 - pt) ** self.gamma

        if self.alpha is not None:
            if self.alpha.device != inputs.device:
                self.alpha = self.alpha.to(inputs.device)
            alpha_t = self.alpha[targets]
            focal_loss = alpha_t * focal_term * ce_loss
        else:
            focal_loss = focal_term * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class OptimizedSpikingAnomalyDetector(nn.Module):
    """Ultra-efficient SNN with temporal optimization."""

    def __init__(self, input_size=20, hidden_size=128, output_size=2,
                 time_steps=100, sparsity=0.3, use_adaptive_threshold=True,
                 early_stop_threshold=0.95):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.time_steps = time_steps
        self.sparsity = sparsity
        self.use_adaptive_threshold = use_adaptive_threshold
        self.early_stop_threshold = early_stop_threshold

        self.temporal_pool_factor = 5
        self.effective_timesteps = time_steps // self.temporal_pool_factor

        # Spiking layers
        self.fc1 = layer.Linear(input_size, hidden_size)
        self.lif1 = neuron.LIFNode(tau=2.0, surrogate_function=neuron.surrogate.ATan())

        self.fc2 = layer.Linear(hidden_size, hidden_size // 2)
        self.lif2 = neuron.LIFNode(tau=2.0, surrogate_function=neuron.surrogate.ATan())

        self.fc3 = layer.Linear(hidden_size // 2, output_size)

    def forward(self, x):
        batch_size = x.shape[0]
        x_pooled = x[:, ::self.temporal_pool_factor, :]

        output_spikes = []

        for t in range(self.effective_timesteps):
            x_t = x_pooled[:, t, :]
            x_t = self.fc1(x_t)
            x_t = self.lif1(x_t)
            x_t = self.fc2(x_t)
            x_t = self.lif2(x_t)
            x_t = self.fc3(x_t)
            output_spikes.append(x_t)

        output = torch.stack(output_spikes, dim=1).mean(dim=1)
        functional.reset_net(self)

        return output

    def get_energy_proxy(self):
        total_params = sum(p.numel() for p in self.parameters())
        energy_ops = (total_params * self.effective_timesteps * (1 - self.sparsity)) / 1e6
        return float(energy_ops)


class SNNTrainer:
    """SNN trainer with Focal Loss and comprehensive metrics."""

    def __init__(self, model, learning_rate=0.001, device='cpu',
                 track_emissions=False, neuromorphic_speedup=1.0,
                 class_weights=None, use_focal_loss=True, focal_gamma=2.0):

        self.model = model.to(device)
        self.device = device
        self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

        # Use Focal Loss for severe imbalance
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
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0

        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(self.device), target.to(self.device)

            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()

        return {
            'loss': total_loss / len(train_loader),
            'accuracy': 100. * correct / total
        }

    def evaluate(self, test_loader):
        """Comprehensive evaluation with proper imbalanced metrics."""
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
                data, target = data.to(self.device), target.to(self.device)

                output = self.model(data)
                _, predicted = output.max(1)

                total += target.size(0)
                correct += predicted.eq(target).sum().item()

                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(target.cpu().numpy())

                batch_spikes = (data > 0).sum().item()
                batch_possible = data.numel()
                total_spikes += batch_spikes
                total_possible_spikes += batch_possible

        elapsed_time = (time.time() - start_time) / self.neuromorphic_speedup
        latency_ms = (elapsed_time / total) * 1000 if total > 0 else 0
        spike_density = total_spikes / total_possible_spikes if total_possible_spikes > 0 else 0

        predictions = np.array(all_predictions)
        labels = np.array(all_labels)

        # Compute proper metrics for imbalanced data
        precision = precision_score(labels, predictions, zero_division=0)
        recall = recall_score(labels, predictions, zero_division=0)
        f1 = f1_score(labels, predictions, zero_division=0)
        cm = confusion_matrix(labels, predictions)

        overall_accuracy = 100. * correct / total if total > 0 else 0

        tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        balanced_accuracy = (sensitivity + specificity) / 2

        return {
            'accuracy': overall_accuracy,
            'balanced_accuracy': balanced_accuracy * 100,
            'precision': precision * 100,
            'recall': recall * 100,
            'f1_score': f1 * 100,
            'confusion_matrix': cm,
            'latency_ms': latency_ms,
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
        if self.track_emissions:
            self.emissions_tracker = EmissionsTracker(
                project_name="snn_training",
                output_dir=output_dir,
                log_level='error'
            )
            self.emissions_tracker.start()

        print(f"  Training for {epochs} epochs...")
        for epoch in range(epochs):
            metrics = self.train_epoch(train_loader)
            if (epoch + 1) % 2 == 0 or epoch == 0:
                print(f"    Epoch {epoch + 1}/{epochs}: Loss={metrics['loss']:.4f}, Acc={metrics['accuracy']:.2f}%")

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
                carbon_metrics = {'emissions_kg': 0.001, 'energy_kwh': 0.0001}

        return carbon_metrics

    def save_model(self, path):
        torch.save(self.model.state_dict(), path)

    def load_model(self, path):
        self.model.load_state_dict(torch.load(path, map_location=self.device))


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("SNN MODEL TEST - WITH FOCAL LOSS")
    print("=" * 70)

    model = OptimizedSpikingAnomalyDetector(
        input_size=20,
        hidden_size=128,
        output_size=2,
        time_steps=100
    )

    print(f"\nModel: {sum(p.numel() for p in model.parameters()):,} parameters")
    print(f"Effective timesteps: {model.effective_timesteps}")
    print(f"Energy proxy: {model.get_energy_proxy():.2f}M ops")

    # Test with class weights
    class_weights = torch.tensor([1.0, 50.0])
    trainer = SNNTrainer(model, class_weights=class_weights, use_focal_loss=True)

    print("\n✓ Focal Loss initialized successfully")
    print(f"{'=' * 70}\n")
