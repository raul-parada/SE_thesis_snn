# batch_processor.py
"""
Batch processor for multiple Loghub datasets.
Searches in ./data/ folder and processes all structured log files.
"""

import os
import glob
import json
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime


class BatchProcessor:
    """Process multiple log datasets from data/ directory."""

    def __init__(self, data_dir="./data", output_dir="logs/batch_results"):
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.results = {}

    def discover_datasets(self):
        """Find all Loghub CSV files - EXCLUDE template files."""
        print(f"\n{'=' * 60}")
        print("DATASET DISCOVERY")
        print(f"{'=' * 60}")
        print(f"Searching in: {self.data_dir.absolute()}")

        # Multiple patterns to catch different naming conventions
        patterns = [
            "*_structured_corrected.csv",
            "*_structured*.csv",
            "*.log_structured*.csv",
            "*2k*.csv"
        ]

        datasets = []

        # Search in data/ directory
        if self.data_dir.exists():
            for pattern in patterns:
                datasets.extend(glob.glob(str(self.data_dir / pattern)))

        # Also search in root (for backward compatibility)
        for pattern in patterns:
            datasets.extend(glob.glob(pattern))

        # Remove duplicates and FILTER OUT template files
        datasets = sorted(list(set(datasets)))
        datasets = [d for d in datasets if 'template' not in Path(d).stem.lower()]

        print(f"Found {len(datasets)} dataset(s):")
        for i, ds in enumerate(datasets, 1):
            file_size = Path(ds).stat().st_size / 1024  # KB
            print(f"  {i}. {Path(ds).name} ({file_size:.1f} KB)")

        if not datasets:
            print("\n[WARNING] No datasets found!")
            print("Expected location: ./data/*_structured_corrected.csv")
            print("Or: ./*_structured_corrected.csv")

        return datasets

    def process_dataset(self, filepath, window_size=10, stride=5, max_len=20):
        """Process a single dataset through the pipeline."""
        from data_loader import LogDataLoader
        from spike_encoder import SpikeEncoder, EncodingStrategy

        dataset_name = Path(filepath).stem

        print(f"\n{'=' * 70}")
        print(f"PROCESSING: {dataset_name}")
        print(f"{'=' * 70}")

        try:
            loader = LogDataLoader(filepath)
            loader.load()
            schema = loader.auto_detect_schema()

            sequences, labels = loader.get_sequences(window_size=window_size, stride=stride)
            normalized = loader.normalize_for_snn(sequences, max_len=max_len)

            encoder = SpikeEncoder(strategy=EncodingStrategy.RATE, time_steps=100)
            spike_trains = encoder.encode(normalized)

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

            if len(labels) > 0:
                stats['normal_count'] = int(np.sum(labels == 0))
                stats['anomaly_count'] = int(np.sum(labels == 1))
                stats['anomaly_ratio'] = float(stats['anomaly_count'] / len(labels) * 100)

            if hasattr(loader, 'anomalous_event_ids'):
                stats['anomalous_event_ids'] = list(loader.anomalous_event_ids)

            self.results[dataset_name] = {
                'stats': stats,
                'sequences': normalized,
                'labels': labels,
                'spike_trains': spike_trains
            }

            print(f"\n✓ Successfully processed {dataset_name}")
            return stats

        except Exception as e:
            print(f"\n✗ Error processing {dataset_name}: {str(e)}")
            import traceback
            traceback.print_exc()
            return None

    def generate_summary_report(self):
        """Generate comprehensive summary report."""
        if not self.results:
            print("[WARNING] No results to report.")
            return

        print(f"\n{'=' * 60}")
        print("GENERATING SUMMARY REPORT")
        print(f"{'=' * 60}")

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

        df = pd.DataFrame(summary_data)

        csv_path = self.output_dir / 'batch_summary.csv'
        df.to_csv(csv_path, index=False)
        print(f"\n✓ Summary saved to: {csv_path}")
        print(f"\n{df.to_string(index=False)}")

        json_data = {
            'processing_date': datetime.now().isoformat(),
            'total_datasets': len(self.results),
            'datasets': {name: result['stats'] for name, result in self.results.items()}
        }

        json_path = self.output_dir / 'batch_summary.json'
        with open(json_path, 'w') as f:
            json.dump(json_data, f, indent=2, default=str)
        print(f"✓ Detailed report saved to: {json_path}")

        return df


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("BATCH PROCESSOR - DATASET DISCOVERY TEST")
    print("=" * 70)

    processor = BatchProcessor()
    datasets = processor.discover_datasets()

    if datasets:
        print(f"\n✓ Ready to process {len(datasets)} datasets")
        print("\nTo run full pipeline, use:")
        print("  python run_pipeline.py")
    else:
        print("\n✗ No datasets found")
        print("  Add *_structured_corrected.csv files to ./data/ folder")

    print(f"\n{'=' * 70}")
    print("TEST COMPLETE")
    print(f"{'=' * 70}")
