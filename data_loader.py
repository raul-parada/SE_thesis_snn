"""
Log Data Loader with EventID-Based Anomaly Detection

This module provides comprehensive log data loading and preprocessing functionality
for anomaly detection systems. It supports automatic schema detection, template-based
anomaly identification using EventID analysis, and adaptive sequence generation.

Key features:
    - Automatic delimiter detection for various CSV formats
    - Template file discovery and analysis for anomaly detection
    - Keyword-based anomaly identification with fallback strategies
    - Adaptive threshold adjustment based on dataset characteristics
    - Sequence generation with sliding window approach
    - Label normalization for binary classification
"""

import pandas as pd
import numpy as np
import re
import glob
import sys
import random
from pathlib import Path
from collections import Counter


class LogDataLoader:
    """
    Comprehensive log data loader with intelligent anomaly detection.

    This class handles loading structured log files, detecting anomalies based on
    event templates, and preparing data for machine learning models. It includes
    multiple fallback strategies to handle diverse log formats from the Loghub dataset.

    Attributes:
        filepath (str): Path to the structured log file
        sample_ratio (float): Proportion of data to sample (0.0 to 1.0)
        df (pd.DataFrame): Loaded dataframe
        label_col (str): Name of the label column
        timestamp_col (str): Name of the timestamp column
        content_col (str): Name of the content column
        original_label_type (str): Type of labels in the dataset
        ground_truth_loaded (bool): Whether ground truth labels were loaded
        anomalous_event_ids (set): Set of EventIDs classified as anomalies
        template_file (str): Path to the template file if found
        ERROR_KEYWORDS (list): Comprehensive list of error-indicating keywords
    """

    def __init__(self, filepath: str, sample_ratio: float = 1.0):
        """
        Initialize the log data loader.

        Args:
            filepath (str): Path to the structured log CSV file
            sample_ratio (float): Proportion of data to use (0.0 to 1.0). Default: 1.0
        """
        self.filepath = filepath
        self.sample_ratio = sample_ratio
        self.df = None
        self.label_col = None
        self.timestamp_col = None
        self.content_col = None
        self.original_label_type = None
        self.ground_truth_loaded = False
        self.anomalous_event_ids = set()
        self.template_file = self._find_template_file(filepath)

        # Comprehensive error keywords for anomaly detection
        self.ERROR_KEYWORDS = [
            'error', 'err', 'exception', 'fatal', 'critical', 'fail', 'failed', 'failure',
            'crash', 'abort', 'timeout', 'denied', 'refused', 'unreachable', 'unavailable',
            'invalid', 'corrupt', 'malformed', 'violation', 'forbidden', 'unauthorized',
            'killed', 'terminated', 'deadlock', 'overflow', 'leak', 'dump', 'down',
            'warn', 'warning', 'alert', 'reject', 'disconnect', 'hung', 'stuck',
            'lost', 'missing', 'unable', 'cannot', 'could not', 'retrying', 'retry',
            'panic', 'severe', 'emergency'
        ]

    def _find_template_file(self, structured_file):
        """
        Locate the corresponding template file for the structured log file.

        This method searches for template files using various naming patterns
        in multiple directories to support different Loghub dataset conventions.

        Args:
            structured_file (str): Path to the structured log file

        Returns:
            str: Path to template file if found, None otherwise
        """
        base_path = Path(structured_file)
        filename = base_path.stem

        # Generate possible template file name patterns
        patterns = [
            filename.replace('_structured_corrected', '_templates_corrected'),
            filename.replace('_structured', '_templates'),
            filename.split('_structured')[0] + '_templates_corrected'
        ]

        # Search in multiple directory locations
        search_dirs = [base_path.parent, base_path.parent / 'data', Path('.') / 'data']

        for search_dir in search_dirs:
            if not search_dir.exists():
                continue

            for pattern in patterns:
                template_path = search_dir / (pattern + '.csv')
                if template_path.exists():
                    return str(template_path)

        return None

    def load(self):
        """
        Load the structured log file with automatic delimiter detection.

        This method attempts to load the CSV file using various delimiters
        (comma, tab, space) to handle different formatting conventions.
        If a template file exists, it performs template-based anomaly analysis.

        Returns:
            pd.DataFrame: Loaded log dataframe

        Raises:
            ValueError: If the file cannot be loaded with any delimiter
        """
        print(f"\n{'=' * 60}")
        print(f"LOADING: {Path(self.filepath).name}")
        print(f"{'=' * 60}")

        # Try multiple delimiters
        for delimiter in [',', '\t', ' ']:
            try:
                df = pd.read_csv(self.filepath, delimiter=delimiter, on_bad_lines='skip')
                if len(df.columns) > 1:
                    self.df = df
                    break
            except Exception:
                continue

        if self.df is None:
            raise ValueError(f"Failed to load {self.filepath}")

        # Perform template analysis if template file exists
        if self.template_file:
            self._analyze_event_templates()

        # Apply sampling if specified
        if self.sample_ratio < 1.0:
            self.df = self.df.sample(frac=self.sample_ratio, random_state=42)

        print(f"Loaded: {len(self.df)} entries, {len(self.df.columns)} columns")

        return self.df

    def _analyze_event_templates(self):
        """
        Analyze event templates to identify anomalous EventIDs.

        This method uses a multi-strategy approach:
        1. Keyword matching: Searches for error-indicating keywords in templates
        2. Statistical fallback: Uses occurrence frequency for rare event detection
        3. Random fallback: Randomly samples events if other methods fail

        The method creates ground truth labels based on identified anomalous EventIDs.
        """
        try:
            df_template = pd.read_csv(self.template_file)

            print(f"\n[TEMPLATE ANALYSIS]")
            print(f"  Template file: {Path(self.template_file).name}")
            print(f"  Template rows: {len(df_template)}")

            # Identify EventID and Template columns
            eventid_col = None
            template_col = None

            for col in df_template.columns:
                col_lower = col.lower()
                if 'eventid' in col_lower or col_lower == 'id':
                    eventid_col = col
                elif 'template' in col_lower or 'content' in col_lower:
                    template_col = col

            if not eventid_col or not template_col:
                print(f"  Warning: Required columns not found")
                print(f"  Available: {df_template.columns.tolist()}")
                return

            print(f"  Using columns: EventId='{eventid_col}', Template='{template_col}'")

            # Analyze templates for error keywords
            anomalous_events = []

            for idx, row in df_template.iterrows():
                event_id = row[eventid_col]
                template_text = str(row[template_col]).lower()

                # Check for error patterns in template
                for keyword in self.ERROR_KEYWORDS:
                    if keyword in template_text:
                        anomalous_events.append({
                            'EventId': event_id,
                            'Keyword': keyword,
                            'Template': template_text[:60]
                        })
                        self.anomalous_event_ids.add(event_id)
                        break

            print(f"  Anomalous EventIDs found: {len(self.anomalous_event_ids)}")

            # Display examples of detected anomalies
            if anomalous_events and len(anomalous_events) <= 5:
                print(f"  Examples:")
                for i, event in enumerate(anomalous_events[:5], 1):
                    print(f"    {i}. {event['EventId']}: '{event['Template']}...' [{event['Keyword']}]")

            # Fallback strategy if no anomalies detected via keywords
            if len(self.anomalous_event_ids) == 0:
                print(f"\n  Warning: No anomalies detected with error keywords")
                print(f"  Sample templates (first 5):")

                for i in range(min(5, len(df_template))):
                    sample = str(df_template.iloc[i][template_col])[:80]
                    eid = df_template.iloc[i][eventid_col]
                    print(f"    {i + 1}. [{eid}] {sample}...")

                # Statistical fallback: Use rare templates
                if 'Occurrences' in df_template.columns:
                    # Mark templates in bottom 10th percentile as anomalies
                    threshold = df_template['Occurrences'].quantile(0.1)
                    rare_templates = df_template[df_template['Occurrences'] <= threshold]
                    self.anomalous_event_ids.update(rare_templates[eventid_col].tolist())

                    print(f"\n  [FALLBACK] Using rare templates (≤{threshold:.0f} occurrences)")
                    print(f"  Marked {len(self.anomalous_event_ids)} rare EventIDs as anomalies")

                else:
                    # Last resort: Random sampling
                    print(f"\n  [FALLBACK] Marking 10% of templates as anomalies (random)")
                    num_anomalies = max(1, len(df_template) // 10)
                    random.seed(42)
                    random_ids = random.sample(list(df_template[eventid_col]), num_anomalies)
                    self.anomalous_event_ids.update(random_ids)
                    print(f"  Marked {len(self.anomalous_event_ids)} EventIDs as anomalies")

            # Create ground truth labels based on identified anomalous EventIDs
            if 'EventId' in self.df.columns and self.anomalous_event_ids:
                self.df['GroundTruthLabel'] = self.df['EventId'].apply(
                    lambda x: 1 if x in self.anomalous_event_ids else 0
                )

                self.label_col = 'GroundTruthLabel'
                self.original_label_type = 'eventid_based'
                self.ground_truth_loaded = True

                # Calculate label distribution
                normal_count = sum(self.df['GroundTruthLabel'] == 0)
                anomaly_count = sum(self.df['GroundTruthLabel'] == 1)

                print(f"  Labels: {normal_count} normal ({normal_count / len(self.df) * 100:.1f}%), "
                      f"{anomaly_count} anomaly ({anomaly_count / len(self.df) * 100:.1f}%)")
            else:
                if 'EventId' not in self.df.columns:
                    print(f"  Warning: EventId column not found in structured dataset")
                else:
                    print(f"  Warning: No anomalous EventIDs identified")

        except Exception as e:
            print(f"  Error: Template analysis failed: {e}")
            import traceback
            traceback.print_exc()

    def intelligent_label_detection(self):
        """
        Detect and identify the label column in the dataset.

        Prioritizes ground truth labels created from template analysis,
        then falls back to existing label columns in the data.

        Returns:
            str: Name of the detected label column, or None if not found
        """
        if self.ground_truth_loaded and 'GroundTruthLabel' in self.df.columns:
            self.label_col = 'GroundTruthLabel'
            self.original_label_type = 'eventid_based'
            return self.label_col

        return None

    def normalize_labels(self):
        """
        Normalize labels to binary format (0 for normal, 1 for anomaly).

        This method handles various label formats including EventID-based labels,
        string labels, and numeric labels.

        Returns:
            np.ndarray: Normalized binary labels, or None if no labels available
        """
        if self.label_col is None or self.label_col not in self.df.columns:
            return None

        labels = self.df[self.label_col].copy()

        # Handle EventID-based labels
        if self.original_label_type == 'eventid_based':
            normalized = labels.values

        # Handle string labels
        elif labels.dtype == 'object':
            normalized = labels.apply(
                lambda x: 1 if any(kw in str(x).lower() for kw in ['anomal', 'error', 'warn']) else 0
            ).values

        # Handle binary numeric labels
        elif set(labels.unique()) <= {0, 1}:
            normalized = labels.values

        # Handle other numeric labels
        else:
            normalized = labels.apply(lambda x: 0 if x == min(labels.unique()) else 1).values

        # Ensure numpy array format
        if not isinstance(normalized, np.ndarray):
            normalized = np.array(normalized)

        return normalized

    def auto_detect_schema(self):
        """
        Automatically detect the schema of the log dataset.

        This method identifies content, label, and timestamp columns using
        heuristics based on common naming conventions.

        Returns:
            dict: Schema dictionary with detected column names and metadata
        """
        # Detect content column
        content_candidates = ['eventtemplate', 'content', 'message', 'text']

        for col in self.df.columns:
            if any(c in col.lower() for c in content_candidates):
                self.content_col = col
                break

        # Fallback to first object column if not found
        if self.content_col is None:
            for col in self.df.columns:
                if self.df[col].dtype == 'object':
                    self.content_col = col
                    break

        # Detect label column
        self.intelligent_label_detection()

        # Detect timestamp column
        timestamp_candidates = ['time', 'timestamp', 'date', 'datetime']

        for col in self.df.columns:
            if any(c in col.lower() for c in timestamp_candidates):
                self.timestamp_col = col
                break

        return {
            'label': self.label_col,
            'timestamp': self.timestamp_col,
            'content': self.content_col,
            'ground_truth': self.ground_truth_loaded
        }

    def tokenize_content(self, max_vocab=5000):
        """
        Tokenize log content into numeric sequences.

        This method extracts words from log content, builds a vocabulary,
        and converts text to sequences of token IDs.

        Args:
            max_vocab (int): Maximum vocabulary size. Default: 5000

        Returns:
            tuple: (tokenized sequences, vocabulary dictionary)

        Raises:
            ValueError: If no content column is found
        """
        # Ensure content column is identified
        if self.content_col is None or self.content_col not in self.df.columns:
            for col in self.df.columns:
                if self.df[col].dtype == 'object':
                    self.content_col = col
                    break

        if self.content_col is None:
            raise ValueError("No content column found for tokenization")

        # Prepare content with null handling
        content = self.df[self.content_col].fillna('').astype(str)

        # Extract all tokens from content
        all_tokens = []
        for text in content:
            tokens = re.findall(r'\w+', text.lower())
            all_tokens.extend(tokens)

        # Handle empty token case
        if not all_tokens:
            all_tokens = ['unknown'] * len(content)

        # Build vocabulary from most common tokens
        token_counts = Counter(all_tokens)
        vocab = {token: idx + 1 for idx, (token, _) in enumerate(token_counts.most_common(max_vocab))}
        vocab[''] = 0  # Reserve 0 for padding

        # Convert content to token sequences
        tokenized = []
        for text in content:
            tokens = re.findall(r'\w+', text.lower())

            # Handle empty content
            if not tokens:
                tokens = ['']

            token_ids = [vocab.get(token, 0) for token in tokens]
            tokenized.append(token_ids)

        return tokenized, vocab

    def get_sequences(self, window_size=10, stride=1, anomaly_threshold='adaptive'):
        """
        Generate sequences using sliding window approach with adaptive thresholding.

        This method creates overlapping sequences from tokenized logs and assigns
        binary labels based on the proportion of anomalies within each window.
        The threshold adapts based on the dataset's anomaly characteristics.

        Args:
            window_size (int): Number of log entries per sequence. Default: 10
            stride (int): Step size between windows. Default: 1
            anomaly_threshold (str or float): Threshold for labeling sequences.
                'adaptive' automatically adjusts based on dataset. Default: 'adaptive'

        Returns:
            tuple: (sequences array, labels array)
        """
        print(f"\n{'=' * 60}")
        print("SEQUENCE GENERATION")
        print(f"{'=' * 60}")

        # Tokenize content
        tokenized, vocab = self.tokenize_content()
        normalized_labels = self.normalize_labels()

        if normalized_labels is None:
            print("  Warning: No labels available")
            return np.array([], dtype=object), np.array([])

        # Calculate log-level anomaly ratio
        log_anomaly_ratio = sum(normalized_labels == 1) / len(normalized_labels)

        # Adaptive threshold selection based on dataset characteristics
        if anomaly_threshold == 'adaptive':
            if log_anomaly_ratio < 0.05:
                threshold = 0.1
                print(f"  [ADAPTIVE] Low anomalies ({log_anomaly_ratio * 100:.1f}%) → threshold=10%")
            elif log_anomaly_ratio < 0.2:
                threshold = 0.3
                print(f"  [ADAPTIVE] Moderate anomalies ({log_anomaly_ratio * 100:.1f}%) → threshold=30%")
            else:
                threshold = 0.5
                print(f"  [ADAPTIVE] High anomalies ({log_anomaly_ratio * 100:.1f}%) → threshold=50%")
        else:
            threshold = anomaly_threshold
            print(f"  Threshold: {threshold * 100:.0f}%")

        print(f"  Window size: {window_size}, Stride: {stride}")

        # Generate sequences with sliding window
        sequences, labels = [], []

        for i in range(0, len(tokenized) - window_size + 1, stride):
            sequences.append(tokenized[i:i + window_size])

            # Calculate anomaly ratio within window
            window_labels = normalized_labels[i:i + window_size]
            anomaly_ratio = sum(window_labels == 1) / len(window_labels)

            # Assign sequence label based on threshold
            sequence_label = 1 if anomaly_ratio >= threshold else 0
            labels.append(sequence_label)

        if not sequences:
            print("  Warning: No sequences generated")
            return np.array([], dtype=object), np.array([])

        print(f"Generated: {len(sequences)} sequences from {len(tokenized)} logs")

        # Calculate and display sequence-level statistics
        if labels:
            normal = int(sum(1 for l in labels if l == 0))
            anomaly = int(sum(1 for l in labels if l == 1))
            anomaly_pct = anomaly / len(labels) * 100 if len(labels) > 0 else 0

            print(f"Labels: Normal: {normal} ({100 - anomaly_pct:.1f}%), Anomaly: {anomaly} ({anomaly_pct:.1f}%)")

            # Quality checks and warnings
            if anomaly > normal:
                print(f"  Warning: Dataset is highly anomalous ({anomaly_pct:.0f}% anomalies)")
            elif anomaly == 0 and log_anomaly_ratio > 0:
                print(f"  Warning: Threshold too high - scattered anomalies not captured")
            else:
                print(f"  Realistic distribution achieved")

        return np.array(sequences, dtype=object), np.array(labels)

    def normalize_for_snn(self, sequences, max_len=20):
        """
        Normalize sequences for Spiking Neural Network processing.

        This method flattens nested sequences, pads or truncates to a fixed length,
        and handles empty sequences appropriately.

        Args:
            sequences (np.ndarray): Array of sequences to normalize
            max_len (int): Maximum sequence length. Default: 20

        Returns:
            np.ndarray: Normalized sequences of shape (n_sequences, max_len)
        """
        if len(sequences) == 0:
            return np.array([])

        normalized = []

        for seq in sequences:
            # Flatten nested lists
            flat = [token for sublist in seq for token in sublist] if isinstance(seq[0], list) else seq

            # Handle empty sequences
            if not flat:
                flat = [0]

            # Pad or truncate to max_len
            flat = (flat + [0] * max_len)[:max_len]
            normalized.append(flat)

        return np.array(normalized)


def discover_log_files(directory=".", data_subfolder="data"):
    """
    Discover structured log files in the specified directory.

    This function searches for CSV files matching common Loghub naming patterns,
    excluding template files from the results.

    Args:
        directory (str): Base directory to search. Default: "."
        data_subfolder (str): Subdirectory name for data files. Default: "data"

    Returns:
        list: Sorted list of discovered log file paths
    """
    patterns = ["*_structured_corrected.csv", "*_structured*.csv"]
    files = []

    # Search in base directory
    for pattern in patterns:
        files.extend(glob.glob(str(Path(directory) / pattern)))

    # Search in data subdirectory
    data_path = Path(directory) / data_subfolder
    if data_path.exists():
        for pattern in patterns:
            files.extend(glob.glob(str(data_path / pattern)))

    # Filter out template files and remove duplicates
    files = [f for f in files if 'template' not in Path(f).stem.lower()]

    return sorted(list(set(files)))


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("DATA LOADER TEST - Template Analysis & Adaptive Thresholds")
    print("=" * 70)

    # Discover available log files
    log_files = discover_log_files(".")

    if not log_files:
        print("\n[ERROR] No datasets found")
        sys.exit(1)

    print(f"\nFound {len(log_files)} dataset(s)")

    # Process first three datasets as test
    for log_file in log_files[:3]:
        loader = LogDataLoader(log_file)
        loader.load()
        loader.auto_detect_schema()
        sequences, labels = loader.get_sequences(window_size=10, stride=1, anomaly_threshold='adaptive')
        print(f"  Result: {len(sequences)} sequences, {sum(labels)} anomalies")

    print(f"\n{'=' * 70}")
    print("TEST COMPLETE")
    print(f"{'=' * 70}")
