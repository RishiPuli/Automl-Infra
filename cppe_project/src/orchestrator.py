"""
src/orchestrator.py
--------------------
Training Orchestrator for CloudAutoML.

Directs the execution flow and controls the resource boundaries (CPU cores)
passed down to the worker nodes, recording performance metrics to log later.
"""

from src.train_models import (
    train_random_forest, train_svm, train_xgboost, train_logistic_regression,
)
from src.experiment_tracker import setup_experiment, log_model_run

MODEL_PARAMS = {
    'RandomForest':       {'n_estimators': 100, 'max_depth': 5, 'random_state': 42},
    'SVM':                {'kernel': 'rbf', 'C': 1.0, 'gamma': 'scale', 'random_state': 42},
    'XGBoost':            {'n_estimators': 100, 'learning_rate': 0.1, 'max_depth': 4},
    'LogisticRegression': {'C': 1.0, 'solver': 'lbfgs', 'max_iter': 200},
}

def run_all_models(X_train, y_train, dataset_size: str, cpu_limit: int = 2) -> list:
    """Run all workers in sequence given the CPU boundary limit."""
    print(f"Orchestrator: Enforcing Cloud Resource Constraint: Max {cpu_limit} CPUs")
    print("Orchestrator: Starting model workers...")

    results = []

    # RandomForest
    try:
        model, dur, ram, cpus = train_random_forest(X_train, y_train, cpu_cores=cpu_limit)
        results.append({'name': 'RandomForest', 'model': model, 'dur': dur, 
                        'ram': ram, 'cpus': cpus, 'path': 'models/random_forest.pkl'})
    except Exception as e: print(f"[ERROR] RF failed: {e}")

    # SVM
    try:
        model, dur, ram, cpus = train_svm(X_train, y_train, cpu_cores=1)
        results.append({'name': 'SVM', 'model': model, 'dur': dur, 
                        'ram': ram, 'cpus': cpus, 'path': 'models/svm.pkl'})
    except Exception as e: print(f"[ERROR] SVM failed: {e}")

    # XGBoost
    try:
        model, dur, ram, cpus = train_xgboost(X_train, y_train, cpu_cores=cpu_limit)
        results.append({'name': 'XGBoost', 'model': model, 'dur': dur, 
                        'ram': ram, 'cpus': cpus, 'path': 'models/xgboost.pkl'})
    except Exception as e: print(f"[ERROR] XGB failed: {e}")

    # LogisticRegression
    try:
        model, dur, ram, cpus = train_logistic_regression(X_train, y_train, cpu_cores=cpu_limit)
        results.append({'name': 'LogisticRegression', 'model': model, 'dur': dur, 
                        'ram': ram, 'cpus': cpus, 'path': 'models/logistic_regression.pkl'})
    except Exception as e: print(f"[ERROR] LR failed: {e}")

    # Logging Layer
    setup_experiment()
    for r in results:
        try:
            log_model_run(
                model_name=r['name'], model_object=r['model'],
                params=MODEL_PARAMS[r['name']], train_time=r['dur'],
                ram_mb=r['ram'], cpu_cores=r['cpus'],
                model_path=r['path'], dataset_size=dataset_size
            )
        except Exception as e:
            print(f"[WARN] MLflow log failed for {r['name']}: {e}")

    _print_summary(results)
    return results


def _print_summary(results: list) -> None:
    sep = '-' * 80
    print(f"\n{sep}")
    print(f"{'Model':<20} | {'Time (s)':<10} | {'RAM (MB)':<10} | {'CPUs':<5} | {'Saved Path':<25}")
    print(sep)
    for r in results:
        print(f"{r['name']:<20} | {r['dur']:<10.2f} | {r['ram']:<10.2f} | "
              f"{r['cpus']:<5} | {r['path']:<25}")
    print(f"{sep}\nTotal succeeded: {len(results)}\n")
