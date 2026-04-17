# tests/conftest.py
"""
Shared pytest fixtures available to all test modules.
"""

import sys
import os

# Ensure project root is on the import path regardless of where pytest is run from
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification


@pytest.fixture(scope="session")
def small_clf_df():
    """
    Small synthetic classification DataFrame (100 rows, 5 numeric features + target).
    Available to every test file via function-argument injection.
    """
    X, y = make_classification(
        n_samples=100,
        n_features=5,
        n_informative=4,
        n_redundant=1,
        n_classes=2,
        random_state=42,
    )
    cols = [f"feature_{i+1}" for i in range(5)]
    df = pd.DataFrame(X, columns=cols)
    df["target"] = y
    return df


@pytest.fixture(scope="session")
def small_clf_arrays(small_clf_df):
    """
    Pre-split X_train, X_test, y_train, y_test derived from small_clf_df.
    All arrays are numpy ndarrays ready for model.fit() / model.predict().
    """
    from sklearn.model_selection import train_test_split

    X = small_clf_df.drop(columns=["target"]).values
    y = small_clf_df["target"].values
    return train_test_split(X, y, test_size=0.2, random_state=42)
