"""
core/hyperparameter_tuner.py
-----------------------------
Optuna Hyperparameter Optimization for CloudAutoML.

Uses TPE (Tree-structured Parzen Estimator) with cross-validated
objectives. Each model gets its own independent Optuna study running
inside its training thread — no shared state, no DB conflicts.
"""

import optuna
import numpy as np
from sklearn.model_selection import cross_val_score, StratifiedKFold, KFold
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression, Ridge
from xgboost import XGBClassifier, XGBRegressor
from utils.logger import get_logger

optuna.logging.set_verbosity(optuna.logging.WARNING)
log = get_logger("HPO")

# Trials per complexity tier
N_TRIALS = {"small": 30, "medium": 20, "large": 10}
CV_FOLDS = 3


# ─── Objectives ───────────────────────────────────────────────────────────────

def _clf_objective(trial, name, X, y, cv):
    if name == "LogisticRegression":
        model = LogisticRegression(
            C=trial.suggest_float("C", 1e-3, 10.0, log=True),
            solver=trial.suggest_categorical("solver", ["saga", "lbfgs"]),
            max_iter=500, random_state=42,
        )
    elif name == "RandomForest":
        model = RandomForestClassifier(
            n_estimators=trial.suggest_int("n_estimators", 50, 300),
            max_depth=trial.suggest_int("max_depth", 3, 15),
            min_samples_split=trial.suggest_int("min_samples_split", 2, 10),
            random_state=42,
        )
    elif name == "XGBoost":
        model = XGBClassifier(
            n_estimators=trial.suggest_int("n_estimators", 50, 300),
            learning_rate=trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            max_depth=trial.suggest_int("max_depth", 3, 9),
            subsample=trial.suggest_float("subsample", 0.6, 1.0),
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.6, 1.0),
            random_state=42, eval_metric="mlogloss", verbosity=0,
        )
    elif name == "SVM":
        model = SVC(
            C=trial.suggest_float("C", 0.01, 100.0, log=True),
            gamma=trial.suggest_categorical("gamma", ["scale", "auto"]),
            kernel=trial.suggest_categorical("kernel", ["rbf", "poly"]),
            random_state=42, probability=True,
        )
    else:
        return 0.0
    return cross_val_score(model, X, y, cv=cv, scoring="accuracy", n_jobs=1).mean()


def _reg_objective(trial, name, X, y, cv):
    if name == "LinearRegression":
        model = Ridge(alpha=trial.suggest_float("alpha", 1e-4, 100.0, log=True))
    elif name == "RandomForestRegressor":
        model = RandomForestRegressor(
            n_estimators=trial.suggest_int("n_estimators", 50, 300),
            max_depth=trial.suggest_int("max_depth", 3, 15),
            min_samples_split=trial.suggest_int("min_samples_split", 2, 10),
            random_state=42,
        )
    elif name == "XGBoostRegressor":
        model = XGBRegressor(
            n_estimators=trial.suggest_int("n_estimators", 50, 300),
            learning_rate=trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            max_depth=trial.suggest_int("max_depth", 3, 9),
            subsample=trial.suggest_float("subsample", 0.6, 1.0),
            random_state=42, verbosity=0,
        )
    else:
        return 0.0
    return cross_val_score(model, X, y, cv=cv, scoring="r2", n_jobs=1).mean()


# ─── Public API ───────────────────────────────────────────────────────────────

def tune_hyperparameters(
    model_name: str, task_type: str,
    X_train: np.ndarray, y_train: np.ndarray,
    complexity: str, emit=None,
) -> dict:
    """Run Optuna TPE study for one model. Returns best_params dict."""
    n_trials = N_TRIALS.get(complexity, 20)

    if task_type == "classification":
        cv  = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=42)
        obj = lambda t: _clf_objective(t, model_name, X_train, y_train, cv)
    else:
        cv  = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=42)
        obj = lambda t: _reg_objective(t, model_name, X_train, y_train, cv)

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42),
    )
    study.optimize(obj, n_trials=n_trials, show_progress_bar=False)

    best = study.best_params
    val  = study.best_value
    if emit:
        emit(f"  🔬 HPO [{model_name}] best_cv={val:.4f} | {best}")
    log.info("[HPO] %s: best_cv=%.4f | %s", model_name, val, best)
    return best


def build_tuned_model(model_name: str, task_type: str, params: dict, cpu_cores: int):
    """Instantiate a model using Optuna-tuned hyperparameters."""
    p = params
    if task_type == "classification":
        if model_name == "LogisticRegression":
            return LogisticRegression(
                C=p.get("C", 1.0), solver=p.get("solver", "saga"),
                max_iter=500, random_state=42, n_jobs=cpu_cores,
            )
        if model_name == "RandomForest":
            return RandomForestClassifier(
                n_estimators=p.get("n_estimators", 100),
                max_depth=p.get("max_depth", 8),
                min_samples_split=p.get("min_samples_split", 2),
                random_state=42, n_jobs=cpu_cores,
            )
        if model_name == "XGBoost":
            return XGBClassifier(
                n_estimators=p.get("n_estimators", 100),
                learning_rate=p.get("learning_rate", 0.1),
                max_depth=p.get("max_depth", 5),
                subsample=p.get("subsample", 1.0),
                colsample_bytree=p.get("colsample_bytree", 1.0),
                random_state=42, eval_metric="mlogloss", verbosity=0, n_jobs=cpu_cores,
            )
        if model_name == "SVM":
            return SVC(
                C=p.get("C", 1.0), gamma=p.get("gamma", "scale"),
                kernel=p.get("kernel", "rbf"), random_state=42, probability=True,
            )
    else:
        if model_name == "LinearRegression":
            return Ridge(alpha=p.get("alpha", 1.0))
        if model_name == "RandomForestRegressor":
            return RandomForestRegressor(
                n_estimators=p.get("n_estimators", 100),
                max_depth=p.get("max_depth", 8),
                min_samples_split=p.get("min_samples_split", 2),
                random_state=42, n_jobs=cpu_cores,
            )
        if model_name == "XGBoostRegressor":
            return XGBRegressor(
                n_estimators=p.get("n_estimators", 100),
                learning_rate=p.get("learning_rate", 0.1),
                max_depth=p.get("max_depth", 5),
                subsample=p.get("subsample", 1.0),
                random_state=42, verbosity=0, n_jobs=cpu_cores,
            )
    return None
