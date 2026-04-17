"""
src/experiment_tracker.py
--------------------------
MLflow Tracking Layer with Resource Awareness.

Logs hyperparameters, training time, CPU usage, RAM footprint, and model artifacts.
"""

import os
import mlflow
import mlflow.sklearn

def setup_experiment(experiment_name: str = 'CloudAutoML_Experiments') -> str:
    mlflow.set_experiment(experiment_name)
    print(f"MLflow experiment set to: {experiment_name}")
    return experiment_name


def log_model_run(model_name: str, model_object, params: dict, 
                  train_time: float, ram_mb: float, cpu_cores: int, 
                  model_path: str, dataset_size: str) -> str:
    """Logs metrics, artifacts, and resource constraints."""
    
    with mlflow.start_run(run_name=f"{model_name}_{dataset_size}"):
        
        # Log model structural parameters
        mlflow.log_params(params)
        
        # Log metadata parameters
        mlflow.log_param('dataset_size', dataset_size)
        
        # Log RESOURCE metrics
        mlflow.log_metric('train_time_seconds', train_time)
        mlflow.log_metric('ram_used_mb', ram_mb)
        mlflow.log_metric('cpu_cores_allocated', cpu_cores)

        # Log artifact
        if os.path.exists(model_path):
            mlflow.log_artifact(model_path)

        try:
            mlflow.sklearn.log_model(model_object, model_name)
        except Exception:
            pass  # XGBoost has its own flavor; safely catch any warning

        run_id = mlflow.active_run().info.run_id

    print(f"MLflow logged: [{model_name} | {ram_mb:.2f}MB RAM | {cpu_cores} CPUs]")
    return run_id


def get_all_runs(experiment_name: str = 'CloudAutoML_Experiments'):
    runs_df = mlflow.search_runs(experiment_names=[experiment_name])
    print(f"\nTotal MLflow runs found for '{experiment_name}': {len(runs_df)}")
    return runs_df
