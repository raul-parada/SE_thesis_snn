"""
Comprehensive Evaluation Framework for Anomaly Detection Models

This module provides a complete evaluation framework for assessing anomaly detection
models across multiple dimensions including detection performance, software engineering
quality metrics, and detailed anomaly reporting.

Key components:
    - AnomalyDetectionReport: Detailed analysis and reporting of detection results
    - EngineeringMetricsEvaluator: Software quality metrics (LoC, MI, CC)
    - PerformanceEvaluator: Detection performance and runtime metrics
    - Type conversion utilities for JSON serialization
"""

import os
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from radon.complexity import cc_visit
from radon.metrics import mi_visit
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from datetime import datetime


def convert_to_native_types(obj):
    """
    Recursively convert numpy/pandas types to Python native types for JSON serialization.

    This function handles nested structures (dicts, lists, tuples) and converts
    numpy/pandas types to standard Python types that can be serialized to JSON.

    Args:
        obj: Object to convert (can be dict, list, tuple, numpy type, pandas type, etc.)

    Returns:
        Converted object with native Python types
    """
    if isinstance(obj, dict):
        return {k: convert_to_native_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_native_types(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_to_native_types(item) for item in obj)
    elif isinstance(obj, (np.integer, np.int64, np.int32, np.int16, np.int8)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32, np.float16)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, (pd.Series, pd.DataFrame)):
        return obj.to_dict()
    else:
        return obj


class AnomalyDetectionReport:
    """
    Generator for detailed anomaly detection reports and analysis.

    This class provides comprehensive reporting functionality for anomaly detection
    results, including detection summaries, confusion matrices, performance metrics,
    and detailed listings of detected anomalies with ground truth comparisons.
    """

    @staticmethod
    def analyze_predictions(predictions, labels, dataset_name, loader=None, sequences=None):
        """
        Analyze and display comprehensive anomaly detection results.

        This method generates a detailed report including detection summaries,
        ground truth comparisons, confusion matrix, detection metrics, and
        examples of detected anomalies.

        Args:
            predictions (np.ndarray): Model predictions (0=normal, 1=anomaly)
            labels (np.ndarray): True labels (0=normal, 1=anomaly)
            dataset_name (str): Name of the dataset being analyzed
            loader (LogDataLoader, optional): Data loader for accessing original logs
            sequences (np.ndarray, optional): Sequence data for display

        Returns:
            dict: Summary statistics including total sequences, predicted anomalies,
                  predicted normal, and anomaly ratio
        """
        print(f"\n{'=' * 70}")
        print(f"ANOMALY DETECTION REPORT: {dataset_name}")
        print(f"{'=' * 70}")

        # Calculate detection summary statistics
        total = len(predictions)
        predicted_anomalies = int(sum(predictions))
        predicted_normal = int(total - predicted_anomalies)

        print(f"\n[DETECTION SUMMARY]")
        print(f"  Total sequences: {total}")
        print(f"  Predicted NORMAL: {predicted_normal:>5} ({predicted_normal / total * 100:>5.1f}%)")
        print(f"  Predicted ANOMALY: {predicted_anomalies:>5} ({predicted_anomalies / total * 100:>5.1f}%)")

        # Analyze anomaly ratio and provide warnings if unusual
        anomaly_ratio = predicted_anomalies / total * 100

        if anomaly_ratio > 50:
            print(f"\n  WARNING: {anomaly_ratio:.1f}% anomalies detected!")
            print(f"  More anomalies than normal is unusual.")
            print(f"  Possible causes:")
            print(f"    - Dataset is highly anomalous")
            print(f"    - Model threshold needs adjustment")
            print(f"    - Check label quality")
        elif anomaly_ratio < 1:
            print(f"\n  INFO: Very few anomalies ({anomaly_ratio:.1f}%)")
            print(f"  System appears very stable.")
        else:
            print(f"\n  Anomaly ratio appears normal ({anomaly_ratio:.1f}%)")

        # Display ground truth comparison if labels are available
        if labels is not None and len(labels) > 0:
            true_anomalies = int(sum(labels))
            true_normal = int(len(labels) - true_anomalies)

            print(f"\n[GROUND TRUTH]")
            print(f"  True NORMAL: {true_normal:>5} ({true_normal / len(labels) * 100:>5.1f}%)")
            print(f"  True ANOMALY: {true_anomalies:>5} ({true_anomalies / len(labels) * 100:>5.1f}%)")

            # Calculate confusion matrix components
            tp = int(sum((predictions == 1) & (labels == 1)))
            fp = int(sum((predictions == 1) & (labels == 0)))
            tn = int(sum((predictions == 0) & (labels == 0)))
            fn = int(sum((predictions == 0) & (labels == 1)))

            print(f"\n[CONFUSION MATRIX]")
            print(f"                    Predicted Normal  Predicted Anomaly")
            print(f"  True Normal       {tn:>6}            {fp:>6} (false alarms)")
            print(f"  True Anomaly      {fn:>6}            {tp:>6} (detected)")

            # Calculate detection metrics
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

            print(f"\n[DETECTION METRICS]")
            print(f"  Precision: {precision * 100:>5.1f}% (accuracy of anomaly predictions)")
            print(f"  Recall:    {recall * 100:>5.1f}% (percentage of anomalies detected)")
            print(f"  F1-Score:  {f1 * 100:>5.1f}% (harmonic mean)")

            # Provide warnings for concerning metrics
            if fn > 0:
                print(f"\n  WARNING: {fn} anomalies MISSED (false negatives)")
            if fp > tp * 2:
                print(f"  WARNING: High false alarm rate ({fp} false positives)")

        # Display detected anomalies
        AnomalyDetectionReport._display_detected_anomalies(
            predictions, labels, loader, sequences
        )

        return {
            'total_sequences': total,
            'predicted_anomalies': predicted_anomalies,
            'predicted_normal': predicted_normal,
            'anomaly_ratio': anomaly_ratio
        }

    @staticmethod
    def _display_detected_anomalies(predictions, labels, loader=None, sequences=None):
        """
        Display sample detected anomalies with details.

        Shows up to 10 detected anomalies with their sequence index, ground truth
        status (if available), and original log content (if loader is provided).

        Args:
            predictions (np.ndarray): Model predictions
            labels (np.ndarray): True labels
            loader (LogDataLoader, optional): Data loader for accessing logs
            sequences (np.ndarray, optional): Sequence data
        """
        anomaly_indices = np.where(predictions == 1)[0]

        if len(anomaly_indices) == 0:
            print(f"\n{'=' * 70}")
            print("  NO ANOMALIES DETECTED - System appears normal!")
            print(f"{'=' * 70}")
            return

        print(f"\n{'=' * 70}")
        print(f"DETECTED ANOMALIES (Showing top 10 of {len(anomaly_indices)})")
        print(f"{'=' * 70}")

        for i, idx in enumerate(anomaly_indices[:10], 1):
            print(f"\n[ANOMALY #{i}] Sequence Index: {idx}")

            # Show ground truth status if available
            if labels is not None and idx < len(labels):
                is_true_anomaly = labels[idx] == 1
                status = "TRUE POSITIVE" if is_true_anomaly else "FALSE POSITIVE"
                print(f"  Status: {status}")

            # Display original log content if available
            if loader is not None and hasattr(loader, 'df') and loader.content_col:
                window_start = idx
                window_end = min(idx + 3, len(loader.df))

                if window_start < len(loader.df):
                    # Show severity if available
                    if loader.label_col and loader.label_col in loader.df.columns:
                        severity = loader.df[loader.label_col].iloc[window_start]
                        print(f"  Severity: {severity}")

                    # Show log entries
                    print(f"  Log content:")
                    log_entries = loader.df[loader.content_col].iloc[window_start:window_end]

                    for j, entry in enumerate(log_entries, 1):
                        # Truncate long entries
                        entry_str = str(entry)[:80]
                        if len(str(entry)) > 80:
                            entry_str += "..."
                        print(f"    [{j}] {entry_str}")

            # Fallback to sequence display if loader not available
            elif sequences is not None and idx < len(sequences):
                print(f"  Sequence: {sequences[idx][:10]}...")

        # Indicate if more anomalies exist
        if len(anomaly_indices) > 10:
            print(f"\n  ... and {len(anomaly_indices) - 10} more anomalies")
            print(f"  (Full list saved in evaluation report)")


class EngineeringMetricsEvaluator:
    """
    Evaluator for software engineering quality metrics.

    This class analyzes code quality metrics including lines of code, maintainability
    index, and cyclomatic complexity to assess the engineering quality of models.

    Attributes:
        project_dir (Path): Base directory for the project
        results (dict): Dictionary storing evaluation results for each model
    """

    def __init__(self, project_dir="."):
        """
        Initialize the engineering metrics evaluator.

        Args:
            project_dir (str): Base directory for the project. Default: "."
        """
        self.project_dir = Path(project_dir)
        self.results = {}

    def count_lines_of_code(self, filepath):
        """
        Count lines of code statistics.

        Analyzes a source file to count total lines, code lines, comment lines,
        and blank lines. This metric is used for integration effort assessment.

        Args:
            filepath (str): Path to the source file

        Returns:
            dict: Dictionary with 'total', 'code', 'comments', and 'blank' counts
        """
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()

        total = len(lines)
        code = sum(1 for line in lines if line.strip() and not line.strip().startswith('#'))
        comments = sum(1 for line in lines if line.strip().startswith('#'))
        blank = total - code - comments

        return {
            'total': int(total),
            'code': int(code),
            'comments': int(comments),
            'blank': int(blank)
        }

    def calculate_maintainability_index(self, filepath):
        """
        Calculate maintainability index using Radon.

        The maintainability index is a metric ranging from 0 to 100 that represents
        the relative ease of maintaining the code. Higher values indicate better
        maintainability.

        Args:
            filepath (str): Path to the source file

        Returns:
            float: Average maintainability index, or 0.0 if calculation fails
        """
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            code = f.read()

        try:
            mi_results = mi_visit(code, multi=True)
            if mi_results:
                return float(np.mean([result.mi for result in mi_results]))
        except:
            pass

        return 0.0

    def calculate_cyclomatic_complexity(self, filepath):
        """
        Calculate cyclomatic complexity metrics.

        Cyclomatic complexity measures the number of linearly independent paths
        through a program's source code. Lower values indicate simpler, more
        maintainable code.

        Args:
            filepath (str): Path to the source file

        Returns:
            dict: Dictionary with 'average', 'max', and 'total' complexity values
        """
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            code = f.read()

        try:
            cc_results = cc_visit(code)
            if cc_results:
                complexities = [result.complexity for result in cc_results]
                return {
                    'average': float(np.mean(complexities)),
                    'max': int(np.max(complexities)),
                    'total': float(np.sum(complexities))
                }
        except:
            pass

        return {'average': 0.0, 'max': 0, 'total': 0.0}

    def evaluate_integration_effort(self, model_type, filepath):
        """
        Perform comprehensive integration effort analysis.

        Calculates and reports all software engineering metrics for a given
        model implementation file.

        Args:
            model_type (str): Name/type of the model
            filepath (str): Path to the model implementation file

        Returns:
            dict: Complete metrics dictionary for the model
        """
        print(f"\n{'=' * 60}")
        print(f"INTEGRATION EFFORT: {model_type}")
        print(f"{'=' * 60}")

        # Calculate all metrics
        loc = self.count_lines_of_code(filepath)
        mi = self.calculate_maintainability_index(filepath)
        cc = self.calculate_cyclomatic_complexity(filepath)

        # Compile results
        results = {
            'model': model_type,
            'lines_of_code': int(loc['code']),
            'lines_comments': int(loc['comments']),
            'maintainability_index': round(float(mi), 2),
            'cyclomatic_complexity_avg': round(float(cc['average']), 2),
            'cyclomatic_complexity_max': int(cc['max'])
        }

        self.results[model_type] = results

        # Display results
        print(f"Lines of Code (LoC):          {loc['code']}")
        print(f"Comment Lines:                {loc['comments']}")
        print(f"Maintainability Index:        {mi:.2f}/100")
        print(f"Avg Cyclomatic Complexity:    {cc['average']:.2f}")
        print(f"Max Cyclomatic Complexity:    {cc['max']}")

        return results

    def generate_comparison_report(self, output_path="logs/evaluation_report.json"):
        """
        Generate comprehensive JSON comparison report.

        Creates a JSON file containing all evaluation results with proper type
        conversion for serialization.

        Args:
            output_path (str): Path for the output JSON file. Default: "logs/evaluation_report.json"

        Returns:
            dict: Results dictionary
        """
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # Convert to serializable types
        results_serializable = convert_to_native_types(self.results)

        # Write JSON report
        with open(output_path, 'w') as f:
            json.dump(results_serializable, f, indent=2)

        print(f"\n[REPORT] Saved to {output_path}")

        return self.results

    def visualize_comparison(self):
        """
        Generate visual comparison charts for engineering metrics.

        Creates a bar chart comparing SNN and Baseline models across key
        software engineering metrics.
        """
        # Verify both models have results
        if 'SNN' not in self.results or 'Baseline' not in self.results:
            print("[WARNING] Need both SNN and Baseline results for comparison")
            return

        # Ensure output directory exists
        os.makedirs("logs", exist_ok=True)

        # Extract metrics for comparison
        metrics = ['lines_of_code', 'maintainability_index', 'cyclomatic_complexity_avg']
        snn_vals = [float(self.results['SNN'][m]) for m in metrics]
        baseline_vals = [float(self.results['Baseline'][m]) for m in metrics]

        # Create bar chart
        x = np.arange(len(metrics))
        width = 0.35

        fig, ax = plt.subplots(figsize=(12, 6))
        ax.bar(x - width / 2, snn_vals, width, label='SNN', color='#2ecc71')
        ax.bar(x + width / 2, baseline_vals, width, label='Baseline ML', color='#3498db')

        # Configure chart
        ax.set_xlabel('Software Engineering Metrics', fontsize=12)
        ax.set_ylabel('Values', fontsize=12)
        ax.set_title('SNN vs Baseline: Engineering Quality Comparison', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(['LoC', 'Maintainability\nIndex', 'Cyclomatic\nComplexity'])
        ax.legend()
        ax.grid(axis='y', alpha=0.3)

        # Save chart
        plt.tight_layout()
        plt.savefig('logs/engineering_comparison.png', dpi=300)
        print("[VISUALIZATION] Saved to logs/engineering_comparison.png")
        plt.close()


class PerformanceEvaluator:
    """
    Evaluator for detection performance and runtime metrics.

    This class calculates and stores performance metrics for anomaly detection
    models, including accuracy, precision, recall, F1-score, latency, and
    energy consumption.

    Attributes:
        results (dict): Dictionary storing evaluation results for each model
    """

    def __init__(self):
        """Initialize the performance evaluator."""
        self.results = {}

    def evaluate_model(self, model_name, predictions, true_labels, latency_ms,
                       energy_proxy=None, loader=None, sequences=None):
        """
        Calculate comprehensive performance metrics with anomaly detection report.

        This method computes all standard classification metrics and generates
        a detailed anomaly detection report including detected anomalies and
        confusion matrix analysis.

        Args:
            model_name (str): Name of the model being evaluated
            predictions (np.ndarray): Model predictions
            true_labels (np.ndarray): Ground truth labels
            latency_ms (float): Inference latency in milliseconds
            energy_proxy (float, optional): Energy consumption metric
            loader (LogDataLoader, optional): Data loader for log content
            sequences (np.ndarray, optional): Original sequences

        Returns:
            dict: Complete evaluation results including metrics and anomaly report
        """
        # Calculate standard classification metrics
        results = {
            'model': model_name,
            'accuracy': float(round(accuracy_score(true_labels, predictions) * 100, 2)),
            'precision': float(round(precision_score(true_labels, predictions, zero_division=0) * 100, 2)),
            'recall': float(round(recall_score(true_labels, predictions, zero_division=0) * 100, 2)),
            'f1_score': float(round(f1_score(true_labels, predictions, zero_division=0) * 100, 2)),
            'latency_ms': float(round(latency_ms, 4))
        }

        # Add energy metric if provided
        if energy_proxy is not None:
            results['energy_proxy'] = float(round(energy_proxy, 2))

        # Generate detailed anomaly detection report
        anomaly_report = AnomalyDetectionReport.analyze_predictions(
            predictions=predictions,
            labels=true_labels,
            dataset_name=model_name,
            loader=loader,
            sequences=sequences
        )

        results['anomaly_detection'] = anomaly_report
        self.results[model_name] = results

        return results


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("EVALUATION FRAMEWORK TEST")
    print("=" * 70)

    # Test anomaly detection report with sample data
    predictions = np.array([0, 0, 1, 0, 1, 0, 0, 0, 1, 0])
    labels = np.array([0, 0, 1, 0, 0, 0, 1, 0, 1, 0])

    AnomalyDetectionReport.analyze_predictions(
        predictions=predictions,
        labels=labels,
        dataset_name="Test Dataset"
    )

    print("\n" + "=" * 70)
    print("TEST COMPLETE")
    print("=" * 70)
