"""
src/data_processing.py
----------------------
Cloud-Native Data Layer for CloudAutoML.

Responsible for:
  - Allowing users to dynamically generate 'small', 'medium', or 'heavy' datasets
    for resource usage experiments without requiring manual CSVs.
  - Checking data quality (nulls, dtypes, stats)
  - Preprocessing (feature/target split, StandardScaler)
  - Train/test splitting with fixed random_state for reproducibility
"""

import pandas as pd
import numpy as np
from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


def load_dataset(size: str = 'small') -> pd.DataFrame:
    """
    Generate synthetic datasets of various sizes for resource experiments.
    
    'small':  1,000 samples, 10 features (Instant, low memory)
    'medium': 50,000 samples, 20 features (Takes a few seconds, higher RAM usage)
    'heavy':  200,000 samples, 40 features (Simulates a real load test for cloud VM)
    
    Returns
    -------
    pd.DataFrame
        Generated dataset with features and a 'target' label column.
    """
    print(f"\n{'='*50}")
    print(f"STEP 1: Generating Dataset - Size: [{size.upper()}]")
    print(f"{'='*50}")
    
    if size == 'small':
        n_samples, n_features = 1000, 10
    elif size == 'medium':
        n_samples, n_features = 50000, 20
    elif size == 'heavy':
        n_samples, n_features = 200000, 40
    else:
        raise ValueError("Invalid size. Use 'small', 'medium', or 'heavy'.")

    # Generate synthetic ML problem
    X, y = make_classification(
        n_samples=n_samples, n_features=n_features, 
        n_classes=3, n_informative=8, random_state=42
    )

    columns = [f"feature_{i+1}" for i in range(n_features)]
    df = pd.DataFrame(X, columns=columns)
    df['target'] = y

    mem_usage_mb = df.memory_usage(deep=True).sum() / (1024 * 1024)

    print(f"Shape: {df.shape}")
    print(f"Memory Footprint: {mem_usage_mb:.2f} MB")
    print("\nFirst 5 rows:")
    print(df.head())

    return df


def check_data_quality(df: pd.DataFrame) -> dict:
    """Perform basic health checks on the dataset."""
    print(f"\n{'='*50}")
    print("STEP 2: Data Quality Report")
    print(f"{'='*50}")

    nulls = int(df.isnull().sum().sum())
    print(f"Total nulls across all columns: {nulls}")

    quality_report = {
        'nulls': nulls,
        'rows': int(df.shape[0]),
        'columns': int(df.shape[1]),
    }
    return quality_report


def preprocess_data(df: pd.DataFrame):
    """Separate target from features, and apply StandardScaler."""
    if df.empty:
        raise ValueError("Dataset is empty. Cannot preprocess.")

    print(f"\n{'='*50}")
    print("STEP 3: Preprocessing (Scaling)")
    print(f"{'='*50}")

    X = df.drop(columns=['target']).values
    y = df['target'].values
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    print(f"Scaled {X.shape[1]} features applied across {X.shape[0]} samples.")
    return X_scaled, y


def split_data(X: np.ndarray, y: np.ndarray, test_size: float = 0.2, random_state: int = 42):
    """Train / Test Split."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    print(f"\n{'='*50}")
    print("STEP 4: Train/Test Split")
    print(f"{'='*50}")
    print(f"Training set: {len(X_train)} samples")
    print(f"Test set:     {len(X_test)} samples")

    return X_train, X_test, y_train, y_test


def run_data_pipeline(dataset_size: str = 'small'):
    """Execute data load, check, preprocess, and split."""
    df = load_dataset(size=dataset_size)
    check_data_quality(df)
    X_scaled, y = preprocess_data(df)
    X_train, X_test, y_train, y_test = split_data(X_scaled, y)

    print(f"\n{'='*50}\nData pipeline complete.\n{'='*50}\n")
    return X_train, X_test, y_train, y_test
