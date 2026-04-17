"""
src/train_models.py
-------------------
Model Training Workers for CloudAutoML.

Each function tracks explicit RAM usage using `psutil` and explicitly restricts
CPU cores using `n_jobs`. Total resources used are returned to the Orchestrator
for MLflow logging.
"""

import os
import time
import joblib
import psutil
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier


def _ensure_models_dir(save_path: str) -> None:
    os.makedirs(save_path, exist_ok=True)


def measure_memory_and_fit(model, X_train, y_train):
    """
    Helper to execute model.fit() while timing it and recording memory.
    Returns: (duration in seconds, peak_ram_used in MB)
    """
    process = psutil.Process(os.getpid())
    mem_before = process.memory_info().rss / (1024 * 1024)  # MB

    start = time.time()
    model.fit(X_train, y_train)
    duration = time.time() - start

    mem_after = process.memory_info().rss / (1024 * 1024)
    ram_used_mb = max(0.01, mem_after - mem_before)  # Minimum 0.01 for visibility

    return duration, ram_used_mb


def train_random_forest(X_train, y_train, save_path='models/', cpu_cores=2):
    _ensure_models_dir(save_path)
    
    model = RandomForestClassifier(n_estimators=100, max_depth=5, 
                                   random_state=42, n_jobs=cpu_cores)
    
    duration, ram_mb = measure_memory_and_fit(model, X_train, y_train)

    model_file = os.path.join(save_path, 'random_forest.pkl')
    joblib.dump(model, model_file)

    print(f"RandomForest  :: {duration:6.2f}s | RAM: {ram_mb:6.2f} MB | CPUs: {cpu_cores}")
    return model, duration, ram_mb, cpu_cores


def train_svm(X_train, y_train, save_path='models/', cpu_cores=1):
    _ensure_models_dir(save_path)
    
    # SVC doesn't support n_jobs native core restrictions, so it uses 1 core.
    model = SVC(kernel='rbf', C=1.0, gamma='scale', 
                random_state=42, probability=True)
    
    duration, ram_mb = measure_memory_and_fit(model, X_train, y_train)

    model_file = os.path.join(save_path, 'svm.pkl')
    joblib.dump(model, model_file)

    print(f"SVM           :: {duration:6.2f}s | RAM: {ram_mb:6.2f} MB | CPUs: 1 (Fixed)")
    return model, duration, ram_mb, 1


def train_xgboost(X_train, y_train, save_path='models/', cpu_cores=2):
    _ensure_models_dir(save_path)
    
    model = XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=4, 
                          random_state=42, eval_metric='mlogloss', verbosity=0,
                          n_jobs=cpu_cores)
    
    duration, ram_mb = measure_memory_and_fit(model, X_train, y_train)

    model_file = os.path.join(save_path, 'xgboost.pkl')
    joblib.dump(model, model_file)

    print(f"XGBoost       :: {duration:6.2f}s | RAM: {ram_mb:6.2f} MB | CPUs: {cpu_cores}")
    return model, duration, ram_mb, cpu_cores


def train_logistic_regression(X_train, y_train, save_path='models/', cpu_cores=2):
    _ensure_models_dir(save_path)
    
    # The lbfgs solver in sklearn correctly supports n_jobs
    model = LogisticRegression(C=1.0, solver='lbfgs', max_iter=200, 
                               random_state=42, n_jobs=cpu_cores)
    
    duration, ram_mb = measure_memory_and_fit(model, X_train, y_train)

    model_file = os.path.join(save_path, 'logistic_regression.pkl')
    joblib.dump(model, model_file)

    print(f"LogisticRegr. :: {duration:6.2f}s | RAM: {ram_mb:6.2f} MB | CPUs: {cpu_cores}")
    return model, duration, ram_mb, cpu_cores
