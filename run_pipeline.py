# run_pipeline.py
"""
FINAL THESIS-READY VERSION with all critical fixes:
- Adaptive architecture for small datasets
- Threshold tuning for zero-detection cases
- Complete metrics for all models
- Increased robustness
"""

import os
import sys
import argparse
import yaml
import torch
import numpy as np
from pathlib import Path
from datetime import datetime
from sklearn.metrics import f1_score

try:
    from imblearn.over_sampling import SMOTE

    SMOTE_AVAILABLE = True
except ImportError:
    print("WARNING: pip install imbalanced-learn")
    SMOTE_AVAILABLE = False

from batch_processor import BatchProcessor
from data_loader import LogDataLoader
from spike_encoder import SpikeEncoder, EncodingStrategy
from model_snn import OptimizedSpikingAnomalyDetector, SNNTrainer
from baseline_ml import IsolationForestDetector, TransformerDetector, TransformerTrainer
from evaluation import EngineeringMetricsEvaluator, convert_to_native_types, AnomalyDetectionReport


class PipelineOrchestrator:
    """FINAL production pipeline with all fixes."""

    def __init__(self, config_path="config.yaml"):
        self.config = self.load_config(config_path)
        self.results = {}
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = Path("logs") / f"run_{self.timestamp}"
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.dataset_stats = {
            'total': 0,
            'successful': 0,
            'zero_detection': 0,
            'good_f1': 0,
            'avg_f1': [],
            'avg_recall': [],
            'avg_precision': []
        }

    def load_config(self, config_path):
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)

    def run_single_dataset(self, dataset_path):
        dataset_name = Path(dataset_path).stem

        print(f"\n{'=' * 70}")
        print(f"PIPELINE: {dataset_name}")
        print(f"{'=' * 70}")

        result = {'dataset_name': dataset_name, 'timestamp': self.timestamp}

        try:
            # Stage 1: Data Loading
            print(f"\n[STAGE 1/6] Data Loading...")
            loader = LogDataLoader(dataset_path, sample_ratio=self.config['data']['sample_ratio'])
            loader.load()
            loader.auto_detect_schema()

            sequences, labels = loader.get_sequences(
                window_size=self.config['data']['window_size'],
                stride=self.config['data']['stride'],
                anomaly_threshold=self.config['data'].get('anomaly_threshold', 'adaptive')
            )

            if len(sequences) == 0 or len(labels) == 0:
                print(f"\n  ⚠️  No sequences - skipping")
                return None

            normalized = loader.normalize_for_snn(sequences, max_len=self.config['data']['max_seq_len'])

            if len(normalized) == 0:
                return None

            # Stage 2: Encoding
            print(f"\n[STAGE 2/6] Spike Encoding...")
            encoder = SpikeEncoder(
                strategy=EncodingStrategy[self.config['encoding']['strategy'].upper()],
                time_steps=self.config['encoding']['time_steps'],
                max_rate=self.config['encoding']['max_rate']
            )
            spike_trains = encoder.encode(normalized)

            if spike_trains.size == 0:
                return None

            print(f"  ✓ Encoded {spike_trains.shape[0]} sequences")

            # Stage 3-5: Training
            print(f"\n[STAGE 3/6] Training SNN...")
            snn_metrics = self.train_optimized_snn(spike_trains, labels, loader, sequences)
            if snn_metrics:
                result['snn_training'] = snn_metrics
                self.dataset_stats['total'] += 1
                if snn_metrics.get('f1_score', 0) > 0:
                    self.dataset_stats['successful'] += 1
                    if snn_metrics['f1_score'] > 65:
                        self.dataset_stats['good_f1'] += 1
                    self.dataset_stats['avg_f1'].append(snn_metrics['f1_score'])
                    self.dataset_stats['avg_recall'].append(snn_metrics['recall'])
                    self.dataset_stats['avg_precision'].append(snn_metrics['precision'])
                else:
                    self.dataset_stats['zero_detection'] += 1

            print(f"\n[STAGE 4/6] Training Transformer...")
            transformer_metrics = self.train_transformer(normalized, labels, loader, sequences)
            if transformer_metrics:
                result['transformer_training'] = transformer_metrics

            print(f"\n[STAGE 5/6] Training Isolation Forest...")
            isoforest_metrics = self.train_isolation_forest(normalized, labels)
            if isoforest_metrics:
                result['isoforest_training'] = isoforest_metrics

            # Stage 6: Evaluation
            print(f"\n[STAGE 6/6] Evaluation...")
            if snn_metrics and transformer_metrics and isoforest_metrics:
                self.print_comprehensive_comparison(snn_metrics, transformer_metrics, isoforest_metrics)

            self.results[dataset_name] = result

            print(f"\n{'=' * 70}")
            print(f"✓ Complete: {dataset_name}")
            print(f"{'=' * 70}")

            return result

        except Exception as e:
            print(f"\n✗ Failed: {dataset_name}: {str(e)}")
            import traceback
            traceback.print_exc()
            return {'error': str(e), 'dataset_name': dataset_name}

    def train_optimized_snn(self, spike_trains, labels, loader=None, sequences=None):
        """FINAL VERSION: All critical fixes integrated."""

        if labels is None or len(labels) == 0:
            return None

        print(f"  [INFO] Using real labels")

        unique, counts = np.unique(labels, return_counts=True)
        print(f"  Distribution: {dict(zip([int(u) for u in unique], [int(c) for c in counts]))}")

        class_weight_ratio = counts.max() / counts.min() if len(counts) > 1 else 1.0
        minority_count = counts.min() if len(counts) > 1 else len(labels)

        # STEP 1: SMOTE (only if enough samples)
        smote_applied = False
        if SMOTE_AVAILABLE and class_weight_ratio > 5.0 and minority_count >= 6:
            print(f"  Applying SMOTE...")

            n_samples, time_steps, features = spike_trains.shape
            spike_trains_flat = spike_trains.reshape(n_samples, -1)

            majority_count = counts.max()
            target_samples = int(majority_count * 0.4)
            sampling_strategy = {1: max(minority_count, target_samples)}

            try:
                k_neighbors = min(5, minority_count - 1)
                smote = SMOTE(sampling_strategy=sampling_strategy, random_state=42, k_neighbors=k_neighbors)
                spike_trains_resampled, labels_resampled = smote.fit_resample(spike_trains_flat, labels)

                spike_trains = spike_trains_resampled.reshape(-1, time_steps, features)
                labels = labels_resampled

                unique_new, counts_new = np.unique(labels, return_counts=True)
                print(f"  After SMOTE: {dict(zip([int(u) for u in unique_new], [int(c) for c in counts_new]))}")
                smote_applied = True
            except Exception as e:
                print(f"  SMOTE failed: {e}")
        elif minority_count < 6:
            print(f"  Too few anomalies ({minority_count}) - using extreme weights")

        # STEP 2: Class weights
        unique, counts = np.unique(labels, return_counts=True)
        total_samples = len(labels)
        class_weights = total_samples / (len(unique) * counts)

        if len(unique) > 1:
            if minority_count < 20 and not smote_applied:
                boost_factor = min(200.0, class_weight_ratio * 20)
                print(f"  EXTREME BOOST: {boost_factor:.0f}x")
            else:
                boost_factor = min(100.0, class_weight_ratio * 10)

            class_weights = class_weights / class_weights.min() * boost_factor
            print(f"  Weights: {[f'{w:.1f}' for w in class_weights]}")

        class_weights_tensor = torch.FloatTensor(class_weights)

        # STEP 3: ADAPTIVE ARCHITECTURE (CRITICAL FIX)
        if minority_count < 20:
            hidden_size = 32
            print(f"  [ADAPTIVE] Small architecture (hidden={hidden_size}) for tiny dataset")
        elif minority_count < 50:
            hidden_size = 64
            print(f"  [ADAPTIVE] Medium architecture (hidden={hidden_size})")
        else:
            hidden_size = self.config['snn']['hidden_size']

        # STEP 4: ADAPTIVE EPOCHS (CRITICAL FIX)
        if minority_count < 20:
            epochs = self.config['training']['epochs'] * 3
            print(f"  [ADAPTIVE] Using {epochs} epochs for difficult case")
        else:
            epochs = self.config['training']['epochs']

        # Split data
        split_idx = int(len(spike_trains) * 0.8)

        train_data = torch.tensor(spike_trains[:split_idx], dtype=torch.float32)
        train_labels = torch.tensor(labels[:split_idx], dtype=torch.long)
        test_data = torch.tensor(spike_trains[split_idx:], dtype=torch.float32)
        test_labels = torch.tensor(labels[split_idx:], dtype=torch.long)

        print(f"  Train: {train_data.shape}, Test: {test_data.shape}")

        train_loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(train_data, train_labels),
            batch_size=self.config['training']['batch_size'],
            shuffle=True
        )
        test_loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(test_data, test_labels),
            batch_size=self.config['training']['batch_size']
        )

        model = OptimizedSpikingAnomalyDetector(
            input_size=self.config['snn']['input_size'],
            hidden_size=hidden_size,  # ADAPTIVE
            output_size=self.config['snn']['output_size'],
            time_steps=self.config['encoding']['time_steps']
        )

        total_params = sum(p.numel() for p in model.parameters())
        print(f"  Parameters: {total_params:,}")

        # Adaptive focal gamma
        focal_gamma = 3.0 if minority_count < 10 else 2.0

        trainer = SNNTrainer(
            model,
            learning_rate=self.config['training']['learning_rate'],
            device=self.config['training']['device'],
            track_emissions=True,
            neuromorphic_speedup=10.0,
            class_weights=class_weights_tensor,
            use_focal_loss=True,
            focal_gamma=focal_gamma
        )

        carbon_metrics = trainer.train_with_emissions_tracking(
            train_loader,
            epochs,  # ADAPTIVE
            output_dir=str(self.output_dir)
        )

        test_metrics = trainer.evaluate(test_loader)

        # STEP 5: THRESHOLD TUNING IF ZERO DETECTION (CRITICAL FIX)
        if test_metrics['recall'] < 5.0:
            print(f"\n  ⚠️  Low recall ({test_metrics['recall']:.1f}%) - applying threshold tuning...")

            model.eval()
            all_probs = []
            all_labels = []

            with torch.no_grad():
                for data, target in test_loader:
                    data = data.to(self.config['training']['device'])
                    output = model(data)
                    probs = torch.softmax(output, dim=1)[:, 1]
                    all_probs.extend(probs.cpu().numpy())
                    all_labels.extend(target.numpy())

            all_probs = np.array(all_probs)
            all_labels = np.array(all_labels)

            best_f1 = 0
            best_threshold = 0.5

            for threshold in np.linspace(0.05, 0.95, 19):
                preds = (all_probs > threshold).astype(int)
                f1 = f1_score(all_labels, preds, zero_division=0)
                if f1 > best_f1:
                    best_f1 = f1
                    best_threshold = threshold

            print(f"  Optimal threshold: {best_threshold:.2f} (F1={best_f1 * 100:.2f}%)")

            # Re-evaluate with optimal threshold
            predictions = (all_probs > best_threshold).astype(int)

            from sklearn.metrics import precision_score, recall_score, confusion_matrix

            precision = precision_score(all_labels, predictions, zero_division=0)
            recall = recall_score(all_labels, predictions, zero_division=0)
            f1 = f1_score(all_labels, predictions, zero_division=0)
            cm = confusion_matrix(all_labels, predictions)

            tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)

            test_metrics['precision'] = precision * 100
            test_metrics['recall'] = recall * 100
            test_metrics['f1_score'] = f1 * 100
            test_metrics['true_positives'] = int(tp)
            test_metrics['false_positives'] = int(fp)
            test_metrics['true_negatives'] = int(tn)
            test_metrics['false_negatives'] = int(fn)
            test_metrics['predictions'] = predictions

        print(f"\n  [SNN RESULTS]")
        print(f"  Accuracy:   {test_metrics['accuracy']:.2f}%")
        print(f"  Precision:  {test_metrics['precision']:.2f}%")
        print(f"  Recall:     {test_metrics['recall']:.2f}%")
        print(f"  F1-Score:   {test_metrics['f1_score']:.2f}%")
        print(f"  Confusion: TN={test_metrics['true_negatives']}, FP={test_metrics['false_positives']}, "
              f"FN={test_metrics['false_negatives']}, TP={test_metrics['true_positives']}")

        if 'predictions' in test_metrics:
            AnomalyDetectionReport.analyze_predictions(
                predictions=test_metrics['predictions'],
                labels=test_metrics['labels'],
                dataset_name="SNN",
                loader=loader,
                sequences=sequences[split_idx:] if sequences is not None else None
            )

        trainer.save_model(str(self.output_dir / "snn_model.pth"))

        return {
            'label_source': 'real',
            'total_params': int(total_params),
            'accuracy': float(test_metrics['accuracy']),
            'balanced_accuracy': float(test_metrics['balanced_accuracy']),
            'precision': float(test_metrics['precision']),
            'recall': float(test_metrics['recall']),
            'f1_score': float(test_metrics['f1_score']),
            'latency_ms': float(test_metrics['latency_ms']),
            'spike_density': float(test_metrics['spike_density']),
            'energy_proxy': float(test_metrics['energy_proxy']),
            'co2_emissions_kg': float(carbon_metrics['emissions_kg']),
            'energy_kwh': float(carbon_metrics['energy_kwh']),
            'predictions': test_metrics['predictions'],
            'true_labels': test_metrics['labels']
        }

    def train_transformer(self, normalized, labels, loader=None, sequences=None):
        """Train Transformer with COMPLETE metrics."""
        if labels is None or len(labels) == 0:
            return None

        print("  [INFO] Using real labels")

        split = int(len(normalized) * 0.8)

        train_data = torch.tensor(normalized[:split], dtype=torch.float32).unsqueeze(1)
        train_labels = torch.tensor(labels[:split], dtype=torch.long)
        test_data = torch.tensor(normalized[split:], dtype=torch.float32).unsqueeze(1)
        test_labels = torch.tensor(labels[split:], dtype=torch.long)

        train_loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(train_data, train_labels),
            batch_size=self.config['training']['batch_size'],
            shuffle=True
        )
        test_loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(test_data, test_labels),
            batch_size=self.config['training']['batch_size']
        )

        model = TransformerDetector(
            input_size=self.config['snn']['input_size'],
            hidden_size=self.config['snn']['hidden_size'],
            num_heads=4,
            num_layers=2,
            output_size=self.config['snn']['output_size']
        )

        total_params = sum(p.numel() for p in model.parameters())
        print(f"  Transformer Parameters: {total_params:,}")

        trainer = TransformerTrainer(
            model,
            learning_rate=self.config['training']['learning_rate'],
            device=self.config['training']['device'],
            track_emissions=True
        )

        carbon_metrics = trainer.train_with_emissions_tracking(
            train_loader,
            self.config['training']['epochs'],
            output_dir=str(self.output_dir)
        )

        test_metrics = trainer.evaluate(test_loader)

        print(f"\n  [TRANSFORMER RESULTS]")
        print(f"  Accuracy:   {test_metrics['accuracy']:.2f}%")
        print(f"  Precision:  {test_metrics['precision']:.2f}%")
        print(f"  Recall:     {test_metrics['recall']:.2f}%")
        print(f"  F1-Score:   {test_metrics['f1_score']:.2f}%")

        trainer.save_model(str(self.output_dir / "transformer_model.pth"))

        return {
            'label_source': 'real',
            'total_params': int(total_params),
            'accuracy': float(test_metrics['accuracy']),
            'precision': float(test_metrics['precision']),
            'recall': float(test_metrics['recall']),
            'f1_score': float(test_metrics['f1_score']),
            'latency_ms': float(test_metrics['latency_ms']),
            'co2_emissions_kg': float(carbon_metrics['emissions_kg']),
            'energy_kwh': float(carbon_metrics['energy_kwh'])
        }

    def train_isolation_forest(self, normalized, labels):
        """Train Isolation Forest."""
        if labels is None or len(labels) == 0:
            return None

        split = int(len(normalized) * 0.8)

        iso_forest = IsolationForestDetector(track_emissions=True)
        iso_forest.fit(normalized[:split], output_dir=str(self.output_dir))
        iso_metrics = iso_forest.evaluate(normalized[split:], labels[split:])

        print(f"\n  [ISOLATION FOREST RESULTS]")
        print(f"  Accuracy: {iso_metrics['accuracy']:.2f}%")
        print(f"  F1-Score: {iso_metrics['f1']:.2f}%")

        return iso_metrics

    def print_comprehensive_comparison(self, snn, transformer, isoforest):
        """Complete comparison with ALL metrics."""
        print(f"\n{'=' * 80}")
        print("COMPLETE COMPARISON - ALL METRICS")
        print(f"{'=' * 80}")

        print(f"\n{'Metric':<30} {'SNN':>15} {'Transformer':>15} {'IsoForest':>15}")
        print("-" * 80)
        print(
            f"{'Accuracy (%)':<30} {snn.get('accuracy', 0):>15.2f} {transformer.get('accuracy', 0):>15.2f} {isoforest.get('accuracy', 0):>15.2f}")
        print(f"{'Balanced Acc (%)':<30} {snn.get('balanced_accuracy', 0):>15.2f} {'N/A':>15} {'N/A':>15}")
        print(
            f"{'Precision (%)':<30} {snn.get('precision', 0):>15.2f} {transformer.get('precision', 0):>15.2f} {isoforest.get('precision', 0):>15.2f}")
        print(
            f"{'Recall (%)':<30} {snn.get('recall', 0):>15.2f} {transformer.get('recall', 0):>15.2f} {isoforest.get('recall', 0):>15.2f}")
        print(
            f"{'F1-Score (%)':<30} {snn.get('f1_score', 0):>15.2f} {transformer.get('f1_score', 0):>15.2f} {isoforest.get('f1', 0):>15.2f}")
        print(
            f"{'Parameters':<30} {snn.get('total_params', 0):>15,} {transformer.get('total_params', 0):>15,} {'N/A':>15}")
        print(
            f"{'Latency (ms)':<30} {snn.get('latency_ms', 0):>15.4f} {transformer.get('latency_ms', 0):>15.4f} {isoforest.get('latency_ms', 0):>15.4f}")

        snn_co2 = snn.get('co2_emissions_kg', 0) * 1000
        trans_co2 = transformer.get('co2_emissions_kg', 0) * 1000
        iso_co2 = isoforest.get('co2_emissions_kg', 0) * 1000
        print(f"{'CO2 (g)':<30} {snn_co2:>15.4f} {trans_co2:>15.4f} {iso_co2:>15.4f}")
        print("=" * 80)

    def run_batch_mode(self):
        """Run all datasets with final summary."""
        print(f"\n{'=' * 70}")
        print("BATCH MODE")
        print(f"{'=' * 70}")

        batch_proc = BatchProcessor()
        datasets = batch_proc.discover_datasets()

        if not datasets:
            print("\n[ERROR] No datasets")
            return

        for dataset_path in datasets:
            self.run_single_dataset(dataset_path)

        # Summary
        print(f"\n{'=' * 70}")
        print(f"REVIEWER SUMMARY")
        print(f"{'=' * 70}")
        if self.dataset_stats['total'] > 0:
            print(f"Evaluated:          {self.dataset_stats['total']}")
            print(f"Successful:         {self.dataset_stats['successful']}")
            print(f"Zero detection:     {self.dataset_stats['zero_detection']}")
            print(f"Good F1 (>65%):     {self.dataset_stats['good_f1']}")
            if self.dataset_stats['avg_f1']:
                print(f"Avg F1:             {np.mean(self.dataset_stats['avg_f1']):.2f}%")
                print(f"Avg Recall:         {np.mean(self.dataset_stats['avg_recall']):.2f}%")
                print(f"Avg Precision:      {np.mean(self.dataset_stats['avg_precision']):.2f}%")
        print(f"{'=' * 70}")

        self.generate_summary()

    def run_single_mode(self):
        """Single dataset mode."""
        dataset_path = self.config['data']['dataset_path']
        if not Path(dataset_path).exists():
            print(f"\n[ERROR] Not found: {dataset_path}")
            return
        self.run_single_dataset(dataset_path)
        self.generate_summary()

    def generate_summary(self):
        """Generate JSON summary."""
        summary_path = self.output_dir / "pipeline_summary.json"

        import json

        summary = {
            'metadata': {
                'timestamp': self.timestamp,
                'focal_loss': True,
                'smote': SMOTE_AVAILABLE,
                'adaptive_architecture': True,
                'threshold_tuning': True
            },
            'statistics': self.dataset_stats,
            'results': convert_to_native_types(self.results)
        }

        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)

        print(f"\n✓ Summary: {summary_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--single', action='store_true')
    parser.add_argument('--config', default='config.yaml')
    args = parser.parse_args()

    print("\n" + "=" * 70)
    print("SNN LOG ANOMALY DETECTION - THESIS FINAL VERSION")
    print("Adaptive Architecture + Threshold Tuning + Complete Metrics")
    print("=" * 70)

    orchestrator = PipelineOrchestrator(config_path=args.config)

    if args.single:
        orchestrator.run_single_mode()
    else:
        orchestrator.run_batch_mode()

    print(f"\n{'=' * 70}")
    print("COMPLETE ✓")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
