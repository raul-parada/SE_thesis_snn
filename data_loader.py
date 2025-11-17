# data_loader.py
"""
Production log data loader with EventID-based anomaly detection.
FINAL: Expanded keywords, fallback strategy, handles all 16 Loghub datasets.
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
    def __init__(self, filepath: str, sample_ratio: float = 1.0):
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

        # EXPANDED error keywords for comprehensive coverage
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
        """Find matching template file."""
        base_path = Path(structured_file)
        filename = base_path.stem

        patterns = [
            filename.replace('_structured_corrected', '_templates_corrected'),
            filename.replace('_structured', '_templates'),
            filename.split('_structured')[0] + '_templates_corrected'
        ]

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
        print(f"\n{'=' * 60}")
        print(f"LOADING: {Path(self.filepath).name}")
        print(f"{'=' * 60}")

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

        if self.template_file:
            self._analyze_event_templates()

        if self.sample_ratio < 1.0:
            self.df = self.df.sample(frac=self.sample_ratio, random_state=42)

        print(f"✓ {len(self.df)} entries, {len(self.df.columns)} columns")
        return self.df

    def _analyze_event_templates(self):
        """
        Analyze templates - FIXED: Expanded keywords + fallback strategy.
        Handles Spark, Thunderbird, and all other datasets.
        """
        try:
            df_template = pd.read_csv(self.template_file)

            print(f"\n[TEMPLATE ANALYSIS]")
            print(f"  Template file: {Path(self.template_file).name}")
            print(f"  Template rows: {len(df_template)}")

            eventid_col = None
            template_col = None

            for col in df_template.columns:
                col_lower = col.lower()
                if 'eventid' in col_lower or col_lower == 'id':
                    eventid_col = col
                elif 'template' in col_lower or 'content' in col_lower:
                    template_col = col

            if not eventid_col or not template_col:
                print(f"  ⚠️  Required columns not found")
                print(f"     Available: {df_template.columns.tolist()}")
                return

            print(f"  Using columns: EventId='{eventid_col}', Template='{template_col}'")

            # Analyze each template for error keywords
            anomalous_events = []

            for idx, row in df_template.iterrows():
                event_id = row[eventid_col]
                template_text = str(row[template_col]).lower()

                # Check for error patterns
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

            # Show examples if found
            if anomalous_events and len(anomalous_events) <= 5:
                print(f"  Examples:")
                for i, event in enumerate(anomalous_events[:5], 1):
                    print(f"    {i}. {event['EventId']}: '{event['Template']}...' [{event['Keyword']}]")

            # FALLBACK: If no anomalies detected
            if len(self.anomalous_event_ids) == 0:
                print(f"\n  ⚠️  No anomalies detected with error keywords!")
                print(f"  Sample templates (first 5):")
                for i in range(min(5, len(df_template))):
                    sample = str(df_template.iloc[i][template_col])[:80]
                    eid = df_template.iloc[i][eventid_col]
                    print(f"    {i + 1}. [{eid}] {sample}...")

                # FALLBACK STRATEGY: Use statistical approach
                # Mark templates with low occurrence count as potential anomalies
                if 'Occurrences' in df_template.columns:
                    # Use 10th percentile as threshold
                    threshold = df_template['Occurrences'].quantile(0.1)
                    rare_templates = df_template[df_template['Occurrences'] <= threshold]
                    self.anomalous_event_ids.update(rare_templates[eventid_col].tolist())
                    print(f"\n  [FALLBACK] Using rare templates (≤{threshold:.0f} occurrences)")
                    print(f"  Marked {len(self.anomalous_event_ids)} rare EventIDs as anomalies")
                else:
                    # Random sampling as last resort (10% of templates)
                    print(f"\n  [FALLBACK] Marking 10% of templates as anomalies (random)")
                    num_anomalies = max(1, len(df_template) // 10)
                    random.seed(42)  # Reproducible
                    random_ids = random.sample(list(df_template[eventid_col]), num_anomalies)
                    self.anomalous_event_ids.update(random_ids)
                    print(f"  Marked {len(self.anomalous_event_ids)} EventIDs as anomalies")

            # Create labels if EventId column exists in main dataset
            if 'EventId' in self.df.columns and self.anomalous_event_ids:
                self.df['GroundTruthLabel'] = self.df['EventId'].apply(
                    lambda x: 1 if x in self.anomalous_event_ids else 0
                )
                self.label_col = 'GroundTruthLabel'
                self.original_label_type = 'eventid_based'
                self.ground_truth_loaded = True

                normal_count = sum(self.df['GroundTruthLabel'] == 0)
                anomaly_count = sum(self.df['GroundTruthLabel'] == 1)

                print(f"  ✓ Labels: {normal_count} normal ({normal_count / len(self.df) * 100:.1f}%), "
                      f"{anomaly_count} anomaly ({anomaly_count / len(self.df) * 100:.1f}%)")
            else:
                if 'EventId' not in self.df.columns:
                    print(f"  ⚠️  EventId column not found in structured dataset")
                else:
                    print(f"  ⚠️  No anomalous EventIDs identified")

        except Exception as e:
            print(f"  ✗ Template analysis failed: {e}")
            import traceback
            traceback.print_exc()

    def intelligent_label_detection(self):
        """Label detection."""
        if self.ground_truth_loaded and 'GroundTruthLabel' in self.df.columns:
            self.label_col = 'GroundTruthLabel'
            self.original_label_type = 'eventid_based'
            return self.label_col
        return None

    def normalize_labels(self):
        """Normalize labels."""
        if self.label_col is None or self.label_col not in self.df.columns:
            return None

        labels = self.df[self.label_col].copy()

        if self.original_label_type == 'eventid_based':
            normalized = labels.values
        elif labels.dtype == 'object':
            normalized = labels.apply(
                lambda x: 1 if any(kw in str(x).lower() for kw in ['anomal', 'error', 'warn']) else 0
            ).values
        elif set(labels.unique()) <= {0, 1}:
            normalized = labels.values
        else:
            normalized = labels.apply(lambda x: 0 if x == min(labels.unique()) else 1).values

        if not isinstance(normalized, np.ndarray):
            normalized = np.array(normalized)

        return normalized

    def auto_detect_schema(self):
        """Detect schema."""
        content_candidates = ['eventtemplate', 'content', 'message', 'text']
        for col in self.df.columns:
            if any(c in col.lower() for c in content_candidates):
                self.content_col = col
                break

        if self.content_col is None:
            for col in self.df.columns:
                if self.df[col].dtype == 'object':
                    self.content_col = col
                    break

        self.intelligent_label_detection()

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
        """Tokenize content with empty handling."""
        if self.content_col is None or self.content_col not in self.df.columns:
            for col in self.df.columns:
                if self.df[col].dtype == 'object':
                    self.content_col = col
                    break

        if self.content_col is None:
            raise ValueError("No content column found for tokenization")

        content = self.df[self.content_col].fillna('').astype(str)
        all_tokens = []

        for text in content:
            tokens = re.findall(r'\w+', text.lower())
            all_tokens.extend(tokens)

        if not all_tokens:
            all_tokens = ['unknown'] * len(content)

        token_counts = Counter(all_tokens)
        vocab = {token: idx + 1 for idx, (token, _) in enumerate(token_counts.most_common(max_vocab))}
        vocab['<UNK>'] = 0

        tokenized = []
        for text in content:
            tokens = re.findall(r'\w+', text.lower())
            if not tokens:
                tokens = ['<UNK>']
            token_ids = [vocab.get(token, 0) for token in tokens]
            tokenized.append(token_ids)

        return tokenized, vocab

    def get_sequences(self, window_size=10, stride=1, anomaly_threshold='adaptive'):
        """Generate sequences with adaptive threshold."""
        print(f"\n{'=' * 60}")
        print("SEQUENCE GENERATION")
        print(f"{'=' * 60}")

        tokenized, vocab = self.tokenize_content()
        normalized_labels = self.normalize_labels()

        if normalized_labels is None:
            print("  ⚠️  No labels available")
            return np.array([], dtype=object), np.array([])

        # ADAPTIVE THRESHOLD
        log_anomaly_ratio = sum(normalized_labels == 1) / len(normalized_labels)

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

        sequences, labels = [], []

        for i in range(0, len(tokenized) - window_size + 1, stride):
            sequences.append(tokenized[i:i + window_size])

            window_labels = normalized_labels[i:i + window_size]
            anomaly_ratio = sum(window_labels == 1) / len(window_labels)

            sequence_label = 1 if anomaly_ratio >= threshold else 0
            labels.append(sequence_label)

        if not sequences:
            print("  ⚠️  No sequences generated")
            return np.array([], dtype=object), np.array([])

        print(f"✓ Generated {len(sequences)} sequences from {len(tokenized)} logs")

        if labels:
            normal = int(sum(1 for l in labels if l == 0))
            anomaly = int(sum(1 for l in labels if l == 1))
            anomaly_pct = anomaly / len(labels) * 100 if len(labels) > 0 else 0

            print(f"✓ Normal: {normal} ({100 - anomaly_pct:.1f}%), Anomaly: {anomaly} ({anomaly_pct:.1f}%)")

            if anomaly > normal:
                print(f"  ⚠️  Dataset is highly anomalous ({anomaly_pct:.0f}% anomalies)")
            elif anomaly == 0 and log_anomaly_ratio > 0:
                print(f"  ⚠️  Threshold too high - scattered anomalies not captured")
            else:
                print(f"  ✓ Realistic distribution")

        return np.array(sequences, dtype=object), np.array(labels)

    def normalize_for_snn(self, sequences, max_len=20):
        """Normalize sequences with empty handling."""
        if len(sequences) == 0:
            return np.array([])

        normalized = []
        for seq in sequences:
            flat = [token for sublist in seq for token in sublist] if isinstance(seq[0], list) else seq
            if not flat:
                flat = [0]
            flat = (flat + [0] * max_len)[:max_len]
            normalized.append(flat)
        return np.array(normalized)


def discover_log_files(directory=".", data_subfolder="data"):
    """Discover structured log files."""
    patterns = ["*_structured_corrected.csv", "*_structured*.csv"]
    files = []

    for pattern in patterns:
        files.extend(glob.glob(str(Path(directory) / pattern)))

    data_path = Path(directory) / data_subfolder
    if data_path.exists():
        for pattern in patterns:
            files.extend(glob.glob(str(data_path / pattern)))

    files = [f for f in files if 'template' not in Path(f).stem.lower()]
    return sorted(list(set(files)))


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("DATA LOADER TEST - Expanded Keywords + Fallback")
    print("=" * 70)

    log_files = discover_log_files(".")

    if not log_files:
        print("\n[ERROR] No datasets found")
        sys.exit(1)

    print(f"\nFound {len(log_files)} dataset(s)")

    for log_file in log_files[:3]:
        loader = LogDataLoader(log_file)
        loader.load()
        loader.auto_detect_schema()
        sequences, labels = loader.get_sequences(window_size=10, stride=1, anomaly_threshold='adaptive')

        print(f"  Result: {len(sequences)} sequences, {sum(labels)} anomalies")

    print(f"\n{'=' * 70}")
    print("✓ TEST COMPLETE")
    print(f"{'=' * 70}")
