"""
main.py  (v2)
-------------
CloudAutoML — Headless CLI Entry Point

Usage:
    python main.py                                  # auto-generates demo dataset
    python main.py --csv data/my.csv
    python main.py --csv data/my.csv --target label_col --test 0.2
    python main.py --csv data/my.csv --no-hpo --no-shap   # fast mode
"""

import argparse
import sys
import os

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import pandas as pd
import numpy  as np

from utils.dataset_analyzer import analyze_dataset, get_recommended_models
from utils.logger           import get_logger
from core.resource_manager  import allocate_resources, get_host_stats
from core.orchestrator      import run_pipeline

log = get_logger("Main")


def _banner(host, flags):
    print("\n" + "═" * 65)
    print("  ☁️  CloudAutoML v2 — Production Pipeline")
    print("═" * 65)
    print(f"  RAM  : {host.total_ram_gb:.1f} GB total")
    print(f"  CPUs : {host.total_cpus} logical | {host.physical_cpus} physical")
    active = [k for k, v in flags.items() if v]
    print(f"  Power: {' · '.join(active) if active else 'none'}")
    print("═" * 65 + "\n")


def run(csv_path, target_hint=None, test_size=0.2,
        enable_hpo=True, enable_shap=True,
        enable_ensemble=True, enable_report=True):

    host = get_host_stats()
    flags = {
        "Optuna HPO":        enable_hpo,
        "SHAP":              enable_shap,
        "Stacking Ensemble": enable_ensemble,
        "PDF Report":        enable_report,
    }
    _banner(host, flags)

    log.info("Loading: %s", csv_path)
    df = pd.read_csv(csv_path)

    analysis   = analyze_dataset(df, target_hint=target_hint)
    complexity = analysis["complexity"]
    task_type  = analysis["task_type"]

    print("── Dataset Analysis ──────────────────────────────────────")
    for k, v in analysis.items():
        if not isinstance(v, (list, dict)):
            print(f"  {k:25s}: {v}")
    print("─" * 60)

    # ── Meta-learning: predict best model before training ─────────────────────
    try:
        from core.stability_predictor import StabilityPredictor
        ranking = StabilityPredictor().fit().predict_ranking(analysis)
        print("\n── StabilityPredictor Ranking ────────────────────────────")
        for rank_i, (mdl_name, score) in enumerate(ranking, 1):
            bar = "█" * int(score * 30)
            print(f"  {rank_i}. {mdl_name:<25s}  {score:.3f}  {bar}")
        print("─" * 60)
    except Exception:
        pass

    allocation  = allocate_resources(complexity)
    model_names = get_recommended_models(task_type, complexity)

    print(f"\n── Resource Allocation ─────────────────────────────────────")
    print(f"  Complexity  : {complexity}")
    print(f"  CPU alloc   : {allocation['cpu_allocated']}")
    print(f"  RAM budget  : {allocation['memory_budget_mb']} MB")
    print(f"  Models      : {', '.join(model_names)}")
    print("─" * 60)

    # ── Performance Optimiser — pre-training strategy plan ────────────────────
    try:
        from core.performance_optimizer import PerformanceOptimiser
        perf = PerformanceOptimiser().fit().recommend(
            analysis, allocation, model_names, hpo_requested=enable_hpo,
        )
        print(f"\n── Performance Plan ──────────────────────────────────────────")
        print(f"  Strategy    : {perf['strategy'].replace('_', ' ')} "
              f"({perf['n_workers']} workers)")
        print(f"  HPO         : "
              f"{'recommended' if perf['hpo_recommended'] else 'auto-disabled (too slow)'}")
        print(f"  Est. par.   : ~{perf['total_est_par']:.0f}s  "
              f"(vs ~{perf['total_est_seq']:.0f}s sequential)")
        for mdl, t in perf["estimated_times"].items():
            print(f"  {mdl:<25s}: ~{t:.0f}s")
        if perf["memory_warnings"]:
            for w in perf["memory_warnings"]:
                print(f"  ⚠ MEM WARN  : {w}")
        print("─" * 60 + "\n")
        # Respect the optimiser's HPO override
        if enable_hpo and not perf["hpo_recommended"]:
            print("  ⚡ HPO auto-disabled by PerformanceOptimiser.")
            enable_hpo = False
    except Exception:
        print()  # non-critical; just continue

    out     = run_pipeline(
        df=df, analysis=analysis,
        model_names=model_names, allocation=allocation,
        test_size=test_size,
        log_fn=lambda msg: print(f"  {msg}"),
        enable_hpo=enable_hpo,
        enable_shap=enable_shap,
        enable_ensemble=enable_ensemble,
        enable_report=enable_report,
    )

    results = out["results"]
    best    = out["best_model"]

    print("\n" + "─" * 85)
    primary = "accuracy" if task_type == "classification" else "r2"
    print(f"{'Model':<22} | {primary.upper():>10} | {'CV Mean':>10} | {'Time(s)':>8} | {'RAM(MB)':>8}")
    print("─" * 85)
    for r in results:
        val = r["metrics"].get(primary, 0)
        cv  = r.get("cv_mean", 0)
        print(f"  {r['name']:<20} | {val:>10.4f} | {cv:>10.4f} | "
              f"{r['train_time']:>8.2f} | {r['peak_ram_mb']:>8.1f}")
    print("─" * 85)

    if best:
        cv_m = best.get("cv_mean", 0)
        cv_s = best.get("cv_std",  0)
        print(f"\n🏆 Best: {best['name']} "
              f"({primary}={best['metrics'].get(primary,0):.4f} | "
              f"CV={cv_m:.4f}±{cv_s:.4f})")

    speedup = out.get("speedup", 1.0)
    print(f"\n⚡ Parallel speedup: {speedup}× faster than sequential")
    print(f"\n📈 MLflow:  mlflow ui   → http://localhost:5000")
    print(f"🌐 API:     uvicorn api.prediction_server:app --port 8000")
    print(f"📂 Models   → models/    Plots → reports/")

    pdf = out.get("pdf_report_path", "")
    if pdf:
        print(f"📄 Report   → {pdf}")
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CloudAutoML v2 CLI")
    parser.add_argument("--csv",         default=None,  help="Path to input CSV")
    parser.add_argument("--target",      default=None,  help="Target column name")
    parser.add_argument("--test",        type=float, default=0.2, help="Test split")
    parser.add_argument("--no-hpo",      action="store_true", help="Disable Optuna HPO")
    parser.add_argument("--no-shap",     action="store_true", help="Disable SHAP")
    parser.add_argument("--no-ensemble", action="store_true", help="Disable stacking ensemble")
    parser.add_argument("--no-report",   action="store_true", help="Disable PDF report")
    args = parser.parse_args()

    if args.csv is None:
        print("⚙️  No CSV supplied — generating synthetic demo dataset ...")
        from sklearn.datasets import make_classification
        X_s, y_s = make_classification(
            n_samples=2000, n_features=12, n_classes=4,
            n_informative=8, random_state=42,
        )
        cols = [f"feature_{i+1}" for i in range(12)]
        demo = pd.DataFrame(X_s, columns=cols)
        demo["target"] = y_s.astype(int)
        os.makedirs("data", exist_ok=True)
        demo_path = "data/demo_dataset.csv"
        demo.to_csv(demo_path, index=False)
        print(f"  Saved demo dataset → {demo_path}\n")
        args.csv = demo_path

    run(
        csv_path=args.csv, target_hint=args.target, test_size=args.test,
        enable_hpo=not args.no_hpo,
        enable_shap=not args.no_shap,
        enable_ensemble=not args.no_ensemble,
        enable_report=not args.no_report,
    )
