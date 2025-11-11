"""
Production Pipeline for SNN-Based Log Anomaly Detection

This module orchestrates the complete anomaly detection pipeline including data loading,
spike encoding, model training (SNN, Transformer, Isolation Forest), evaluation, and
comprehensive software engineering metrics analysis.

Key features:
    - Adaptive architecture scaling based on dataset size
    - SMOTE oversampling for severe class imbalance
    - Threshold tuning for zero-detection cases
    - Focal Loss for imbalanced classification
    - Automated software engineering metrics collection
    - Comprehensive evaluation with multiple models
"""

import os
import sys
import argparse
import yaml
import torch
import numpy as np
import subprocess
import json
from pathlib import Path
from datetime import datetime
from sklearn.metrics import f1_score

try:
    from imblearn.over_sampling import SMOTE
    SMOTE_AVAILABLE = True
except ImportError:
    print("WARNING: SMOTE not available. Install with: pip install imbalanced-learn")
    SMOTE_AVAILABLE = False

from batch_processor import BatchProcessor
from data_loader import LogDataLoader
from spike_encoder import SpikeEncoder, EncodingStrategy
from model_snn import OptimizedSpikingAnomalyDetector, SNNTrainer
from baseline_ml import IsolationForestDetector, TransformerDetector, TransformerTrainer
from evaluation import EngineeringMetricsEvaluator, convert_to_native_types, AnomalyDetectionReport


def analyze_software_engineering_metrics():
    """
    Analyze software engineering metrics using Radon.

    This function calculates lines of code, cyclomatic complexity, and dependency
    counts for all model implementations (SNN, Transformer, Isolation Forest).

    Returns:
        dict: Software engineering metrics for each model including LoC,
              cyclomatic complexity, and external dependencies
    """
    print(f"\n{'=' * 70}")
    print("ANALYZING SOFTWARE ENGINEERING METRICS")
    print(f"{'=' * 70}")

    # Define source files for each model
    files = {
        'SNN': ['model_snn.py', 'spike_encoder.py'],
        'Transformer': ['baseline_ml.py'],
        'IsolationForest': ['baseline_ml.py']
    }

    se_metrics = {}

    for model_name, file_list in files.items():
        print(f"\n  Analyzing {model_name}...")
        total_loc = 0
        total_comments = 0
        total_complexity = []
        max_complexity = 0

        for file_path in file_list:
            if not Path(file_path).exists():
                print(f"  Warning: File not found: {file_path}")
                continue

            # Calculate lines of code using Radon
            try:
                result = subprocess.run(
                    ['radon', 'raw', file_path, '-s'],
                    capture_output=True, text=True, timeout=5
                )

                for line in result.stdout.split('\n'):
                    if 'LOC:' in line:
                        total_loc += int(line.split(':')[1].strip())
                    elif 'Comments:' in line:
                        total_comments += int(line.split(':')[1].strip())

            except FileNotFoundError:
                print(f"  Warning: Radon not installed. Falling back to manual counting")
                total_loc += count_lines_manually(file_path)
            except Exception as e:
                print(f"  Warning: Radon analysis failed for {file_path}: {e}")

            # Calculate cyclomatic complexity
            try:
                result = subprocess.run(
                    ['radon', 'cc', file_path, '-s', '-j'],
                    capture_output=True, text=True, timeout=5
                )

                if result.stdout.strip():
                    complexity_data = json.loads(result.stdout)
                    for file_metrics in complexity_data.values():
                        for func in file_metrics:
                            complexity = func.get('complexity', 0)
                            total_complexity.append(complexity)
                            max_complexity = max(max_complexity, complexity)

            except Exception as e:
                print(f"  Warning: Complexity analysis failed for {file_path}: {e}")

        # Calculate average complexity
        avg_complexity = sum(total_complexity) / len(total_complexity) if total_complexity else 0

        # Store metrics
        se_metrics[model_name] = {
            'model': model_name,
            'lines_of_code': total_loc,
            'lines_comments': total_comments,
            'cyclomatic_complexity_avg': round(avg_complexity, 2),
            'cyclomatic_complexity_max': max_complexity,
            'dependencies': count_dependencies(file_list)
        }

        print(f"  LoC: {total_loc}, Comments: {total_comments}, Complexity: {avg_complexity:.2f}")

    print(f"\n{'=' * 70}")
    return se_metrics


def count_lines_manually(file_path):
    """
    Fallback manual line counter for LoC calculation.

    Args:
        file_path (str): Path to source file

    Returns:
        int: Number of code lines (excluding comments and blank lines)
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        # Count non-empty, non-comment lines
        code_lines = [l for l in lines if l.strip() and not l.strip().startswith('#')]
        return len(code_lines)

    except Exception:
        return 0


def count_dependencies(file_list):
    """
    Count unique external dependencies from import statements.

    Args:
        file_list (list): List of source file paths

    Returns:
        int: Number of unique external dependencies
    """
    dependencies = set()

    # Standard library modules to exclude
    stdlib = {
        'os', 'sys', 'math', 'json', 'pathlib', 'datetime', 'typing',
        'argparse', 'subprocess', 're', 'time', 'random', 'collections',
        'itertools', 'functools', 'abc', 'warnings', 'logging'
    }

    for file_path in file_list:
        if not Path(file_path).exists():
            continue

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line.startswith('import ') or line.startswith('from '):
                        # Extract package name
                        if 'import ' in line:
                            parts = line.replace('from ', '').replace('import ', '').split()
                            if parts:
                                pkg = parts[0].split('.')[0]
                                if pkg not in stdlib and not pkg.startswith('_'):
                                    dependencies.add(pkg)
        except Exception:
            pass

    return len(dependencies)


class PipelineOrchestrator:
    """
    Main orchestrator for the anomaly detection pipeline.

    This class manages the complete workflow including dataset processing,
    model training, evaluation, and result aggregation. It supports both
    single-dataset and batch processing modes with adaptive architecture
    selection and threshold tuning.

    Attributes:
        config (dict): Configuration loaded from YAML file
        results (dict): Training and evaluation results for each dataset
        timestamp (str): Execution timestamp for result tracking
        output_dir (Path): Directory for saving logs and results
        dataset_stats (dict): Aggregate statistics across all datasets
        se_metrics (dict): Software engineering metrics for all models
    """

    def __init__(self, config_path="config.yaml"):
        """
        Initialize the pipeline orchestrator.

        Args:
            config_path (str): Path to YAML configuration file. Default: "config.yaml"
        """
        self.config = self.load_config(config_path)
        self.results = {}
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = Path("logs") / f"run_{self.timestamp}"
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize dataset statistics
        self.dataset_stats = {
            'total': 0,
            'successful': 0,
            'zero_detection': 0,
            'good_f1': 0,
            'avg_f1': [],
            'avg_recall': [],
            'avg_precision': []
        }

        # Collect software engineering metrics at startup
        self.se_metrics = analyze_software_engineering_metrics()

    def load_config(self, config_path):
        """
        Load configuration from YAML file.

        Args:
            config_path (str): Path to configuration file

        Returns:
            dict: Loaded configuration
        """
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)

    def run_single_dataset(self, dataset_path):
        """
        Process a single dataset through the complete pipeline.

        This method executes all six pipeline stages: data loading, spike encoding,
        SNN training, Transformer training, Isolation Forest training, and evaluation.

        Args:
            dataset_path (str): Path to the dataset file

        Returns:
            dict: Complete results including metrics for all models, or None if failed
        """
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
                print(f"\n  Warning: No sequences generated - skipping dataset")
                return None

            normalized = loader.normalize_for_snn(sequences, max_len=self.config['data']['max_seq_len'])

            if len(normalized) == 0:
                return None

            # Stage 2: Spike Encoding
            print(f"\n[STAGE 2/6] Spike Encoding...")
            encoder = SpikeEncoder(
                strategy=EncodingStrategy[self.config['encoding']['strategy'].upper()],
                time_steps=self.config['encoding']['time_steps'],
                max_rate=self.config['encoding']['max_rate']
            )

            spike_trains = encoder.encode(normalized)

            if spike_trains.size == 0:
                return None

            print(f"  Encoded {spike_trains.shape[0]} sequences")

            # Stages 3-5: Model Training
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
            print(f"Complete: {dataset_name}")
            print(f"{'=' * 70}")

            return result

        except Exception as e:
            print(f"\nFailed: {dataset_name}: {str(e)}")
            import traceback
            traceback.print_exc()
            return {'error': str(e), 'dataset_name': dataset_name}

    # Additional methods would continue here following the same professional pattern
    # (train_optimized_snn, train_transformer, train_isolation_forest, etc.)

    def generate_summary(self):
        """
        Generate comprehensive JSON summary with all results and metrics.

        Creates a JSON file containing metadata, software engineering metrics,
        dataset statistics, and detailed results for all processed datasets.
        """
        summary_path = self.output_dir / "pipeline_summary.json"

        summary = {
            'metadata': {
                'timestamp': self.timestamp,
                'focal_loss': True,
                'smote': SMOTE_AVAILABLE,
                'adaptive_architecture': True,
                'threshold_tuning': True
            },
            'software_engineering_metrics': self.se_metrics,
            'statistics': self.dataset_stats,
            'results': convert_to_native_types(self.results)
        }

        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)

        print(f"\nSummary saved: {summary_path}")

        # Save SE metrics separately for easy access
        se_metrics_path = self.output_dir / "software_engineering_metrics.json"

        with open(se_metrics_path, 'w') as f:
            json.dump(self.se_metrics, f, indent=2)

        print(f"SE Metrics saved: {se_metrics_path}")


def main():
    """
    Main entry point for the pipeline.

    Parses command-line arguments and executes the pipeline in either
    single-dataset or batch mode.
    """
    parser = argparse.ArgumentParser(
        description='SNN-based log anomaly detection pipeline'
    )
    parser.add_argument('--single', action='store_true',
                       help='Run single dataset mode (uses config.yaml dataset path)')
    parser.add_argument('--config', default='config.yaml',
                       help='Path to configuration file')

    args = parser.parse_args()

    print(f"\n{'=' * 70}")
    print("SNN LOG ANOMALY DETECTION - PRODUCTION PIPELINE")
    print("Features: Adaptive Architecture | Threshold Tuning | Complete Metrics")
    print(f"{'=' * 70}")

    orchestrator = PipelineOrchestrator(config_path=args.config)

    if args.single:
        orchestrator.run_single_mode()
    else:
        orchestrator.run_batch_mode()

    print(f"\n{'=' * 70}")
    print("PIPELINE COMPLETE")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
