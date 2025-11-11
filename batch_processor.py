"""
Batch Processor for Multiple Log Datasets

This module provides batch processing capabilities for multiple Loghub datasets.
It automatically discovers structured log files in the data directory, processes them
through the spike encoding pipeline, and generates comprehensive summary reports.

The processor supports various log file naming conventions and excludes template files
from processing. Results are saved in both CSV and JSON formats for easy analysis.
"""

import os
import glob
import json
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime


class BatchProcessor:
    """
    Batch processor for handling multiple log datasets.

    This class manages the discovery, processing, and reporting of multiple log datasets.
    It searches for structured CSV files, processes them through the spike encoding pipeline,
    and generates summary reports with statistics and metadata.

    Attributes:
        data_dir (Path): Directory containing input datasets
        output_dir (Path): Directory for saving processing results
        results (dict): Dictionary storing processing results for each dataset
    """

    def __init__(self, data_dir="./data", output_dir="logs/batch_results"):
        """
        Initialize the batch processor.

        Args:
            data_dir (str): Path to directory containing log datasets. Default: "./data"
            output_dir (str): Path to directory for saving results. Default: "logs/batch_results"
        """
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)

        # Create output directory if it doesn't exist
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize results storage
        self.results = {}

    def discover_datasets(self):
        """
        Discover all structured log CSV files in the data directory.

        This method searches for CSV files matching various naming conventions used
        by Loghub datasets. Template files are automatically excluded from the results.

        Returns:
            list: List of file paths to discovered datasets
        """
        print(f"
{'=' * 60}")
        print("DATASET DISCOVERY")
        print(f"{'=' * 60}")
        print(f"Searching in: {self.data_dir.absolute()}")

        # Define search patterns for different naming conventions
        patterns = [
            "*_structured_corrected.csv",
            "*_structured*.csv",
            "*.log_structured*.csv",
            "*2k*.csv"
        ]

        datasets = []

        # Search in the specified data directory
        if self.data_dir.exists():
            for pattern in patterns:
                datasets.extend(glob.glob(str(self.data_dir / pattern)))

        # Also search in root directory for backward compatibility
        for pattern in patterns:
            datasets.extend(glob.glob(pattern))

        # Remove duplicates and filter out template files
        datasets = sorted(list(set(datasets)))
        datasets = [d for d in datasets if 'template' not in Path(d).stem.lower()]

        # Display discovered datasets
        print(f"Found {len(datasets)} dataset(s):")
        for i, ds in enumerate(datasets, 1):
            file_size = Path(ds).stat().st_size / 1024  # Convert to KB
            print(f"  {i}. {Path(ds).name} ({file_size:.1f} KB)")

        # Warn if no datasets found
        if not datasets:
            print("
[WARNING] No datasets found!")
            print("Expected location: ./data/*_structured_corrected.csv")
            print("Or: ./*_structured_corrected.csv")

        return datasets

    def process_dataset(self, filepath, window_size=10, stride=5, max_len=20):
        """
        Process a single dataset through the spike encoding pipeline.

        This method loads a log file, extracts sequences, normalizes the data,
        and encodes it as spike trains. Statistics and results are stored for
        later reporting.

        Args:
            filepath (str): Path to the dataset file
            window_size (int): Size of sliding window for sequence extraction. Default: 10
            stride (int): Step size for sliding window. Default: 5
            max_len (int): Maximum sequence length for normalization. Default: 20

        Returns:
            dict: Statistics dictionary for the processed dataset, or None if processing failed
        """
        from data_loader import LogDataLoader
        from spike_encoder import SpikeEncoder, EncodingStrategy

        dataset_name = Path(filepath).stem

        print(f"
{'=' * 70}")
        print(f"PROCESSING: {dataset_name}")
        print(f"{'=' * 70}")

        try:
            # Load and process the dataset
            loader = LogDataLoader(filepath)
            loader.load()
            schema = loader.auto_detect_schema()

            # Extract sequences and labels
            sequences, labels = loader.get_sequences(window_size=window_size, stride=stride)

            # Normalize sequences for SNN processing
            normalized = loader.normalize_for_snn(sequences, max_len=max_len)

            # Encode as spike trains
            encoder = SpikeEncoder(strategy=EncodingStrategy.RATE, time_steps=100)
            spike_trains = encoder.encode(normalized)

            # Compile statistics
            stats = {
                'dataset_name': dataset_name,
                'filepath': str(filepath),
                'total_logs': int(len(loader.df)),
                'num_sequences': int(len(sequences)),
                'has_labels': bool(len(labels) > 0),
                'label_type': loader.original_label_type if hasattr(loader, 'original_label_type') else 'unknown',
                'ground_truth': loader.ground_truth_loaded if hasattr(loader, 'ground_truth_loaded') else False,
                'timestamp': datetime.now().isoformat()
            }

            # Add label statistics if labels are available
            if len(labels) > 0:
                stats['normal_count'] = int(np.sum(labels == 0))
                stats['anomaly_count'] = int(np.sum(labels == 1))
                stats['anomaly_ratio'] = float(stats['anomaly_count'] / len(labels) * 100)

                if hasattr(loader, 'anomalous_event_ids'):
                    stats['anomalous_event_ids'] = list(loader.anomalous_event_ids)

            # Store results for this dataset
            self.results[dataset_name] = {
                'stats': stats,
                'sequences': normalized,
                'labels': labels,
                'spike_trains': spike_trains
            }

            print(f"
Successfully processed {dataset_name}")
            return stats

        except Exception as e:
            print(f"
Error processing {dataset_name}: {str(e)}")
            import traceback
            traceback.print_exc()
            return None

    def generate_summary_report(self):
        """
        Generate comprehensive summary report for all processed datasets.

        This method creates both CSV and JSON reports containing statistics
        and metadata for all successfully processed datasets. The reports include
        dataset names, log counts, sequence information, label statistics, and
        anomaly ratios.

        Returns:
            pd.DataFrame: Summary dataframe, or None if no results to report
        """
        if not self.results:
            print("[WARNING] No results to report.")
            return

        print(f"
{'=' * 60}")
        print("GENERATING SUMMARY REPORT")
        print(f"{'=' * 60}")

        # Compile summary data from all datasets
        summary_data = []
        for dataset_name, result in self.results.items():
            stats = result['stats']
            summary_data.append({
                'Dataset': dataset_name,
                'Total Logs': stats['total_logs'],
                'Sequences': stats['num_sequences'],
                'Has Labels': stats['has_labels'],
                'Label Type': stats.get('label_type', 'unknown'),
                'Ground Truth': stats.get('ground_truth', False),
                'Anomaly %': f"{stats.get('anomaly_ratio', 0):.1f}%" if 'anomaly_ratio' in stats else 'N/A'
            })

        # Create dataframe and save as CSV
        df = pd.DataFrame(summary_data)
        csv_path = self.output_dir / 'batch_summary.csv'
        df.to_csv(csv_path, index=False)

        print(f"
Summary saved to: {csv_path}")
        print(f"
{df.to_string(index=False)}")

        # Create detailed JSON report
        json_data = {
            'processing_date': datetime.now().isoformat(),
            'total_datasets': len(self.results),
            'datasets': {name: result['stats'] for name, result in self.results.items()}
        }

        json_path = self.output_dir / 'batch_summary.json'
        with open(json_path, 'w') as f:
            json.dump(json_data, f, indent=2, default=str)

        print(f"Detailed report saved to: {json_path}")

        return df


if __name__ == "__main__":
    print("
" + "=" * 70)
    print("BATCH PROCESSOR - DATASET DISCOVERY TEST")
    print("=" * 70)

    # Initialize processor and discover datasets
    processor = BatchProcessor()
    datasets = processor.discover_datasets()

    # Display results and instructions
    if datasets:
        print(f"
Ready to process {len(datasets)} datasets")
        print("
To run full pipeline, use:")
        print("  python run_pipeline.py")
    else:
        print("
No datasets found")
        print("  Add *_structured_corrected.csv files to ./data/ folder")

    print(f"
{'=' * 70}")
    print("TEST COMPLETE")
    print(f"{'=' * 70}")
