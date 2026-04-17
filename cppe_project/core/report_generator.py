"""
core/report_generator.py
-------------------------
Auto-generates a multi-page dark-themed PDF experiment report.
Uses matplotlib's PdfPages backend — no external PDF library needed.

Pages:
  1. Title / summary card
  2. Model performance charts (2×2 grid)
  3. Comparison plots (embedded PNGs)
  4. Per-model detail plots
"""

import os
import datetime
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.backends.backend_pdf import PdfPages
from utils.logger import get_logger

log         = get_logger("ReportGen")
REPORTS_DIR = "reports"

_PALETTE = ["#00d2ff", "#a855f7", "#10b981", "#f59e0b", "#ef4444",
            "#60a5fa", "#fb923c", "#34d399"]


def _ax_dark(ax):
    ax.set_facecolor("#1A1D27")
    ax.tick_params(colors="white", labelsize=8)
    ax.xaxis.label.set_color("white")
    ax.yaxis.label.set_color("white")
    ax.title.set_color("white")
    for sp in ax.spines.values():
        sp.set_edgecolor("#333")
    return ax


def _title_page(pdf, analysis, results, best, task_type, speedup):
    fig = plt.figure(figsize=(12, 10), facecolor="#0F1117")
    fig.text(0.5, 0.87, "☁️  CloudAutoML", ha="center",
             fontsize=40, fontweight="bold", color="#00d2ff")
    fig.text(0.5, 0.80, "Automated Machine Learning — Experiment Report",
             ha="center", fontsize=15, color="#a0aec0")
    fig.text(0.5, 0.74,
             datetime.datetime.now().strftime("Generated: %Y-%m-%d  %H:%M:%S"),
             ha="center", fontsize=10, color="#718096")

    # Divider
    ax_d = fig.add_axes([0.1, 0.71, 0.8, 0.002])
    ax_d.set_facecolor("#00d2ff"); ax_d.axis("off")

    items = [
        ("Task Type",     analysis.get("task_type", "—").title()),
        ("Rows",          f"{analysis.get('rows', 0):,}"),
        ("Features",      str(analysis.get("features", 0))),
        ("Complexity",    analysis.get("complexity", "—").title()),
        ("Models Trained",str(len(results))),
        ("Best Model",    best.get("name", "—") if best else "—"),
        ("Wall-clock (s)",str(speedup.get("wall", "—"))),
        ("Speedup",       f"{speedup.get('speedup', 1.0):.2f}×"),
    ]
    xs, y = [0.12, 0.52], 0.64
    for i, (lbl, val) in enumerate(items):
        x = xs[i % 2]
        fig.text(x,        y, lbl + ":", fontsize=11, color="#718096")
        fig.text(x + 0.20, y, str(val),  fontsize=11, color="white",
                 fontweight="bold")
        if i % 2 == 1:
            y -= 0.063

    pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)


def _metrics_page(pdf, results, task_type):
    primary = "accuracy" if task_type == "classification" else "r2"
    valid   = [r for r in results if primary in r.get("metrics", {})]
    if not valid:
        return

    names  = [r["name"]             for r in valid]
    vals   = [r["metrics"][primary] for r in valid]
    times  = [r["train_time"]       for r in valid]
    rams   = [r["peak_ram_mb"]      for r in valid]

    fig = plt.figure(figsize=(14, 9), facecolor="#0F1117")
    fig.suptitle("📊  Model Performance Summary",
                 fontsize=16, fontweight="bold", color="white", y=0.97)
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.5, wspace=0.38)

    def _sub(pos):
        return _ax_dark(fig.add_subplot(pos))

    # 1. Primary metric
    ax1 = _sub(gs[0, 0])
    bars = ax1.bar(names, vals, color=_PALETTE[:len(names)],
                   edgecolor="#222", width=0.55)
    for b, v in zip(bars, vals):
        ax1.text(b.get_x() + b.get_width() / 2,
                 b.get_height() + max(vals) * 0.02,
                 f"{v:.4f}", ha="center", va="bottom",
                 fontsize=8, color="white", fontweight="bold")
    ax1.set_title(primary.upper(), fontsize=11, fontweight="bold")
    ax1.set_ylim(0, min(1.25, max(vals) * 1.3) if vals else 1)
    ax1.yaxis.grid(True, color="#2d3748", linestyle="--", alpha=0.5)
    ax1.set_axisbelow(True)
    plt.setp(ax1.get_xticklabels(), rotation=25, ha="right", fontsize=7)

    # 2. Training time
    ax2 = _sub(gs[0, 1])
    ax2.bar(names, times, color=_PALETTE[1], edgecolor="#222", width=0.55)
    ax2.set_title("Train Time (s)", fontsize=11, fontweight="bold")
    ax2.yaxis.grid(True, color="#2d3748", linestyle="--", alpha=0.5)
    ax2.set_axisbelow(True)
    plt.setp(ax2.get_xticklabels(), rotation=25, ha="right", fontsize=7)

    # 3. Peak RAM
    ax3 = _sub(gs[1, 0])
    ax3.bar(names, rams, color=_PALETTE[2], edgecolor="#222", width=0.55)
    ax3.set_title("Peak RAM (MB)", fontsize=11, fontweight="bold")
    ax3.yaxis.grid(True, color="#2d3748", linestyle="--", alpha=0.5)
    ax3.set_axisbelow(True)
    plt.setp(ax3.get_xticklabels(), rotation=25, ha="right", fontsize=7)

    # 4. Metrics table
    ax4 = _sub(gs[1, 1])
    ax4.axis("off")
    col_hdrs  = ["Model", primary.upper(), "Time(s)", "RAM(MB)"]
    row_data  = [[n, f"{v:.4f}", f"{t:.2f}", f"{r:.1f}"]
                 for n, v, t, r in zip(names, vals, times, rams)]
    tbl = ax4.table(cellText=row_data, colLabels=col_hdrs,
                    cellLoc="center", loc="center")
    tbl.auto_set_font_size(False); tbl.set_fontsize(8); tbl.scale(1, 1.6)
    for (row, col), cell in tbl.get_celld().items():
        cell.set_facecolor("#1A1D27" if row > 0 else "#0f3460")
        cell.set_text_props(color="white")
        cell.set_edgecolor("#333")
    ax4.set_title("Metrics Table", fontsize=11, fontweight="bold", color="white")

    pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)


def _embed_png(pdf, path: str, title: str = ""):
    if not path or not os.path.exists(path):
        return
    try:
        img = plt.imread(path)
        fig = plt.figure(figsize=(12, 8), facecolor="#0F1117")
        if title:
            fig.suptitle(title, fontsize=12, color="#a0aec0")
        ax = fig.add_axes([0.03, 0.03, 0.94, 0.90])
        ax.imshow(img); ax.axis("off")
        pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)
    except Exception as e:
        log.warning("Could not embed %s: %s", path, e)


def generate_report(
    analysis: dict, allocation: dict,
    results: list, best: dict,
    task_type: str, speedup_info: dict,
) -> str:
    """Generate a full PDF report. Returns file path (empty string on failure)."""
    os.makedirs(REPORTS_DIR, exist_ok=True)
    ts   = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(REPORTS_DIR, f"cloudautoml_report_{ts}.pdf")

    try:
        with PdfPages(path) as pdf:
            _title_page(pdf, analysis, results, best, task_type, speedup_info)
            _metrics_page(pdf, results, task_type)

            # Comparison plots
            for fn in ["metrics_comparison.png", "training_time.png"]:
                _embed_png(pdf, os.path.join(REPORTS_DIR, fn), fn)

            # Per-model plots
            for r in results:
                for p in r.get("plot_paths", []):
                    _embed_png(pdf, p, f"{r['name']} — {os.path.basename(p)}")

            d = pdf.infodict()
            d["Title"]   = "CloudAutoML Experiment Report"
            d["Author"]  = "CloudAutoML Platform v2"
            d["Subject"] = f"{task_type.title()} | {analysis.get('rows', 0)} rows"

        log.info("PDF report → %s", path)
        return path
    except Exception as e:
        log.error("Report generation failed: %s", e)
        return ""
