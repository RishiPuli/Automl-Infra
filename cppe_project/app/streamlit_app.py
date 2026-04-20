"""
app/streamlit_app.py
---------------------
CloudAutoML – Resource-Aware AutoML Platform
Clean light theme | Mobile-responsive | No prediction panel.

Cloud-hardened:
  - All psutil calls wrapped in try/except with fallbacks
  - All file writes use relative paths
  - mlruns/ references wrapped in try/except (missing dir = no crash)
  - No hardcoded absolute paths anywhere
"""

import sys
import os
import traceback

# Add cppe_project/ dir (parent of app/) to sys.path so core/, utils/ resolve
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# Also try repo root as fallback (Streamlit Cloud CWD is repo root)
REPO_ROOT = os.path.abspath(os.path.join(ROOT, ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import io
import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from utils.dataset_analyzer import analyze_dataset, get_recommended_models

# ── Cloud-safe imports ────────────────────────────────────────────────────────
try:
    from core.resource_manager import allocate_resources, get_host_stats
    _rm_ok = True
except Exception:
    _rm_ok = False

_orch_err = ""
try:
    from core.orchestrator import run_pipeline
    _orch_ok = True
except Exception as _e:
    _orch_ok = False
    _orch_err = traceback.format_exc()

# ─────────────────────────────────────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CloudAutoML",
    page_icon="☁️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─────────────────────────────────────────────────────────────────────────────
# CSS — Clean Light SaaS Theme, Mobile First
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

  /* ── Global ── */
  html, body, [class*="css"] {
    font-family: 'Inter', sans-serif !important;
    background-color: #f7f8fa !important;
    color: #1a1d23 !important;
  }

  /* ── Sidebar ── */
  [data-testid="stSidebar"] {
    background: #ffffff !important;
    border-right: 1px solid #e5e7eb !important;
  }
  [data-testid="stSidebar"] .stMarkdown p,
  [data-testid="stSidebar"] label,
  [data-testid="stSidebar"] .stMarkdown {
    color: #374151 !important;
  }
  [data-testid="stSidebar"] h3 {
    color: #16a34a !important;
  }

  /* ── Hero header ── */
  .hero {
    background: linear-gradient(135deg, #16a34a 0%, #22c55e 60%, #4ade80 100%);
    border-radius: 16px;
    padding: 40px 36px 32px 36px;
    margin-bottom: 28px;
    color: #ffffff;
    box-shadow: 0 4px 24px rgba(22,163,74,0.22);
  }
  .hero h1 {
    font-size: clamp(1.6rem, 4vw, 2.4rem);
    font-weight: 800;
    margin: 0 0 8px 0;
    letter-spacing: -0.5px;
  }
  .hero p {
    font-size: clamp(0.85rem, 2vw, 1rem);
    opacity: 0.92;
    margin: 0;
  }

  /* ── Section label ── */
  .section-label {
    font-size: 0.68rem;
    font-weight: 700;
    letter-spacing: 2.5px;
    text-transform: uppercase;
    color: #16a34a;
    border-left: 4px solid #22c55e;
    padding: 5px 0 5px 14px;
    margin: 32px 0 18px 0;
    background: #f0fdf4;
    border-radius: 0 6px 6px 0;
  }

  /* ── Metric cards ── */
  .metric-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(120px, 1fr));
    gap: 12px;
    margin-bottom: 22px;
  }
  .metric-card {
    background: #ffffff;
    border: 1px solid #e5e7eb;
    border-radius: 12px;
    padding: 18px 16px;
    text-align: center;
    box-shadow: 0 1px 4px rgba(0,0,0,0.06);
    transition: transform 0.2s, box-shadow 0.2s, border-color 0.2s;
  }
  .metric-card:hover {
    transform: translateY(-2px);
    box-shadow: 0 4px 16px rgba(22,163,74,0.12);
    border-color: #86efac;
  }
  .metric-card .m-val {
    font-size: clamp(1.3rem, 3vw, 1.8rem);
    font-weight: 700;
    color: #16a34a;
    line-height: 1;
  }
  .metric-card .m-lbl {
    font-size: 0.68rem;
    color: #6b7280;
    margin-top: 6px;
    letter-spacing: 0.5px;
    text-transform: uppercase;
  }

  /* ── Info tiles ── */
  .info-row { display: flex; gap: 10px; flex-wrap: wrap; margin: 14px 0; }
  .info-tile {
    background: #ffffff;
    border: 1px solid #e5e7eb;
    border-radius: 8px;
    padding: 8px 16px;
    font-size: 0.85rem;
    color: #374151;
  }
  .info-tile b { color: #16a34a; }
  .info-tile code {
    background: #f0fdf4;
    color: #16a34a;
    padding: 2px 6px;
    border-radius: 4px;
    font-size: 0.8rem;
  }

  /* ── Tags ── */
  .tag {
    display: inline-block;
    border-radius: 6px;
    padding: 3px 12px;
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.5px;
  }
  .tag-clf  { background: #f0fdf4; color: #16a34a; border: 1px solid #86efac; }
  .tag-reg  { background: #fffbeb; color: #b45309; border: 1px solid #fcd34d; }
  .tag-sm   { background: #eff6ff; color: #1d4ed8; border: 1px solid #93c5fd; }
  .tag-med  { background: #fffbeb; color: #b45309; border: 1px solid #fcd34d; }
  .tag-lg   { background: #fef2f2; color: #dc2626; border: 1px solid #fca5a5; }

  /* ── Allocation block ── */
  .alloc-block {
    background: #ffffff;
    border: 1px solid #e5e7eb;
    border-left: 4px solid #22c55e;
    border-radius: 10px;
    padding: 20px 22px;
    font-size: 0.85rem;
    line-height: 2.1;
    color: #374151;
    margin-bottom: 16px;
    box-shadow: 0 1px 4px rgba(0,0,0,0.05);
  }
  .alloc-block .key { color: #6b7280; }
  .alloc-block .val { color: #16a34a; font-weight: 600; }

  /* ── Best model block ── */
  .best-block {
    background: linear-gradient(135deg, #f0fdf4, #dcfce7);
    border: 1.5px solid #86efac;
    border-radius: 12px;
    padding: 22px 26px;
    margin-bottom: 20px;
    box-shadow: 0 2px 12px rgba(22,163,74,0.1);
  }
  .best-block .b-name {
    font-size: clamp(1rem, 3vw, 1.25rem);
    font-weight: 700;
    color: #15803d;
  }
  .best-block .b-detail {
    font-size: 0.85rem;
    color: #374151;
    margin-top: 10px;
    line-height: 1.9;
  }
  .best-block .b-detail strong { color: #16a34a; }

  /* ── Speedup banner ── */
  .speedup-banner {
    background: linear-gradient(90deg, #eff6ff, #f0f9ff);
    border: 1px solid #bfdbfe;
    border-radius: 10px;
    padding: 16px 22px;
    color: #1d4ed8;
    font-size: 0.9rem;
    margin: 16px 0;
    display: flex;
    align-items: center;
    gap: 18px;
    flex-wrap: wrap;
  }
  .speedup-banner .sp-icon { font-size: 1.5rem; }
  .speedup-banner .sp-val  { font-size: 1.3rem; font-weight: 700; color: #16a34a; }
  .speedup-banner .sp-lbl  { font-size: 0.74rem; color: #6b7280; display: block; }

  /* ── Log window ── */
  .logwin {
    background: #1e1e2e;
    border: 1px solid #e5e7eb;
    border-radius: 8px;
    padding: 14px 16px;
    font-family: 'Courier New', monospace;
    font-size: 0.76rem;
    color: #a3e635;
    max-height: 220px;
    overflow-y: auto;
    white-space: pre-wrap;
    line-height: 1.7;
  }

  /* ── Notice ── */
  .notice {
    background: #fffbeb;
    border: 1px solid #fcd34d;
    border-radius: 8px;
    padding: 10px 16px;
    font-size: 0.84rem;
    color: #92400e;
    margin: 8px 0;
  }

  /* ── MLflow block ── */
  .mlflow-block {
    background: #ffffff;
    border: 1px solid #e5e7eb;
    border-radius: 10px;
    padding: 18px 22px;
    font-size: 0.84rem;
    color: #374151;
    line-height: 2;
  }
  .mlflow-block code {
    background: #f0fdf4;
    color: #16a34a;
    padding: 2px 8px;
    border-radius: 4px;
    font-family: 'Courier New', monospace;
  }

  /* ── Landing box ── */
  .landing-box {
    background: #ffffff;
    border: 2px dashed #d1d5db;
    border-radius: 16px;
    padding: 60px 40px;
    text-align: center;
    color: #9ca3af;
    margin-top: 16px;
  }

  /* ── Streamlit overrides ── */
  .stDataFrame { border: 1px solid #e5e7eb !important; border-radius: 8px; }
  .stButton > button {
    background: #16a34a !important;
    color: #ffffff !important;
    border: 1px solid #16a34a !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    padding: 0.5rem 1.5rem !important;
    transition: all 0.2s;
    font-family: 'Inter', sans-serif !important;
  }
  .stButton > button:hover {
    background: #15803d !important;
    border-color: #15803d !important;
    box-shadow: 0 4px 14px rgba(22,163,74,0.25) !important;
    transform: translateY(-1px);
  }
  .stDownloadButton > button {
    background: #ffffff !important;
    color: #16a34a !important;
    border: 1px solid #86efac !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
  }
  .stTabs [data-baseweb="tab"] {
    color: #6b7280 !important;
    font-weight: 500;
  }
  .stTabs [aria-selected="true"] {
    color: #16a34a !important;
    border-bottom: 2px solid #16a34a !important;
  }
  .stProgress > div > div { background-color: #16a34a !important; }
  div[data-testid="stExpander"] {
    background: #ffffff !important;
    border: 1px solid #e5e7eb !important;
    border-radius: 8px !important;
  }
  div[data-testid="stExpander"] summary {
    color: #374151 !important;
  }
  .stAlert { border-radius: 8px !important; }
  #MainMenu { visibility: hidden; }
  footer    { visibility: hidden; }
  header    { visibility: hidden; }

  /* ── Mobile responsiveness ── */
  @media (max-width: 640px) {
    .hero { padding: 28px 20px 22px 20px; }
    .metric-grid { grid-template-columns: repeat(2, 1fr); }
    .info-row { gap: 6px; }
    .info-tile { font-size: 0.78rem; padding: 6px 12px; }
    .alloc-block { font-size: 0.8rem; padding: 14px 16px; }
    .best-block  { padding: 16px 18px; }
    .speedup-banner { padding: 12px 16px; }
    .section-label { font-size: 0.65rem; }
  }
</style>
""", unsafe_allow_html=True)

# Force Streamlit's own theme to light
st._config.set_option("theme.base", "light")


# ─────────────────────────────────────────────────────────────────────────────
# Session state
# ─────────────────────────────────────────────────────────────────────────────
for k, v in {
    "df": None, "analysis": None, "allocation": None,
    "model_names": None, "result": None,
    "trained": False, "logs": [],
}.items():
    if k not in st.session_state:
        st.session_state[k] = v


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ☁️ CloudAutoML")
    st.caption("Intelligent AutoML Platform")
    st.divider()

    st.markdown("**🖥️ Host Machine**")
    if _rm_ok:
        try:
            host = get_host_stats()
            st.markdown(f"CPU (logical): `{host.total_cpus}`")
            st.markdown(f"CPU (physical): `{host.physical_cpus}`")
            st.markdown(f"RAM: `{host.total_ram_gb:.1f} GB`")
        except Exception:
            st.markdown("_System stats unavailable_")
    else:
        st.markdown("_System stats unavailable_")

    st.divider()
    st.markdown("**⚙️ Settings**")
    target_hint = st.text_input("Target column", placeholder="blank = auto-detect")
    test_split  = st.slider("Test split", 0.10, 0.35, 0.20, step=0.05, format="%.2f")

    st.divider()
    st.markdown("**🔧 Pipeline Options**")
    enable_hpo      = st.checkbox("Optuna HPO",          value=True)
    enable_shap     = st.checkbox("SHAP Explanations",   value=True)
    enable_ensemble = st.checkbox("Stacking Ensemble",   value=True)
    enable_report   = st.checkbox("Generate PDF Report", value=True)

    st.divider()
    if st.button("🔄 Reset", use_container_width=True):
        for k in list(st.session_state.keys()):
            del st.session_state[k]
        st.rerun()


# ─────────────────────────────────────────────────────────────────────────────
# Hero header
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
  <h1>☁️ CloudAutoML Platform</h1>
  <p>Upload any CSV → Auto-analyze → Train multiple models in parallel → Get the best model instantly</p>
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Step 1 — Upload
# ─────────────────────────────────────────────────────────────────────────────
st.markdown('<div class="section-label">Step 1 — Dataset Upload</div>', unsafe_allow_html=True)

uploaded = st.file_uploader("Select a CSV file", type=["csv"], label_visibility="collapsed")

if uploaded is not None:
    try:
        df_new = pd.read_csv(uploaded)
        if df_new.empty:
            st.error("File is empty."); st.stop()
        if (st.session_state["df"] is None or
                not df_new.equals(st.session_state["df"])):
            st.session_state.update({
                "df": df_new, "analysis": None, "allocation": None,
                "model_names": None, "result": None,
                "trained": False, "logs": [],
            })
    except Exception as e:
        st.error(f"Could not read CSV: {e}"); st.stop()


# ─────────────────────────────────────────────────────────────────────────────
# Steps 2-5 — require data
# ─────────────────────────────────────────────────────────────────────────────
if st.session_state["df"] is not None:
    df = st.session_state["df"]

    # ── Step 2: Analysis ──────────────────────────────────────────────────────
    st.markdown('<div class="section-label">Step 2 — Dataset Analysis</div>', unsafe_allow_html=True)

    if st.session_state["analysis"] is None:
        hint = target_hint.strip() or None
        try:
            st.session_state["analysis"] = analyze_dataset(df, target_hint=hint)
        except Exception as e:
            st.error(f"Analysis failed: {e}"); st.stop()

    an         = st.session_state["analysis"]
    task_type  = an["task_type"]
    complexity = an["complexity"]

    mv = f"{an['total_missing']}" if an["missing_values"] else "none"
    st.markdown(f"""
    <div class="metric-grid">
      <div class="metric-card"><div class="m-val">{an['rows']:,}</div><div class="m-lbl">Rows</div></div>
      <div class="metric-card"><div class="m-val">{an['features']}</div><div class="m-lbl">Features</div></div>
      <div class="metric-card"><div class="m-val">{an['numerical']}</div><div class="m-lbl">Numerical</div></div>
      <div class="metric-card"><div class="m-val">{an['categorical']}</div><div class="m-lbl">Categorical</div></div>
      <div class="metric-card"><div class="m-val">{an['n_unique_target']}</div><div class="m-lbl">Target Classes</div></div>
      <div class="metric-card"><div class="m-val">{mv}</div><div class="m-lbl">Missing Values</div></div>
    </div>
    """, unsafe_allow_html=True)

    task_tag  = "clf" if task_type == "classification" else "reg"
    cplx_tag  = {"small": "sm", "medium": "med", "large": "lg"}[complexity]
    st.markdown(
        f'<div class="info-row">'
        f'<div class="info-tile"><b>Target:</b> <code>{an["target_column"]}</code></div>'
        f'<div class="info-tile"><b>Task:</b> <span class="tag tag-{task_tag}">{task_type.upper()}</span></div>'
        f'<div class="info-tile"><b>Complexity:</b> <span class="tag tag-{cplx_tag}">{complexity.upper()}</span></div>'
        f'<div class="info-tile"><b>Memory:</b> {an["memory_mb"]:.2f} MB</div>'
        f'</div>',
        unsafe_allow_html=True,
    )

    if an.get("should_sample"):
        st.markdown(
            f'<div class="notice">⚠️ Dataset has {an["rows"]:,} rows — will be sampled to 5,000 for training.</div>',
            unsafe_allow_html=True,
        )
    if an["missing_values"]:
        with st.expander(f"Missing values in {len(an['cols_with_missing'])} column(s)"):
            st.dataframe(
                pd.DataFrame.from_dict(an["cols_with_missing"], orient="index", columns=["Count"]),
                use_container_width=True,
            )
    with st.expander("Data preview (first 10 rows)"):
        st.dataframe(df.head(10), use_container_width=True)


    # ── Step 3: Resource Allocation ───────────────────────────────────────────
    st.markdown('<div class="section-label">Step 3 — Resource Allocation</div>', unsafe_allow_html=True)

    if st.session_state["allocation"] is None:
        try:
            st.session_state["allocation"] = allocate_resources(complexity)
        except Exception:
            st.session_state["allocation"] = {
                "cpu_allocated": 2, "cpu_min": 1, "cpu_max": 2,
                "memory_budget_mb": 512, "available_cpus": 2,
                "physical_cpus": 1, "total_ram_gb": 0.0, "available_ram_gb": 0.0,
            }
        st.session_state["model_names"] = get_recommended_models(task_type, complexity)

    alloc       = st.session_state["allocation"]
    model_names = st.session_state["model_names"]

    col_a, col_b = st.columns([1, 1])
    with col_a:
        st.markdown(f"""
        <div class="alloc-block">
          <div><span class="key">Dataset complexity  </span><span class="val">{complexity}</span></div>
          <div><span class="key">CPU allocated       </span><span class="val">{alloc['cpu_allocated']} cores</span></div>
          <div><span class="key">Memory budget       </span><span class="val">{alloc['memory_budget_mb']} MB</span></div>
          <div><span class="key">Parallel workers    </span><span class="val">{alloc['cpu_allocated']} (ThreadPoolExecutor)</span></div>
          <div><span class="key">Host CPU (logical)  </span><span class="val">{alloc['available_cpus']}</span></div>
          <div><span class="key">Host RAM            </span><span class="val">{alloc['total_ram_gb']:.1f} GB</span></div>
        </div>
        """, unsafe_allow_html=True)
    with col_b:
        st.markdown("**🤖 Models Selected**")
        for i, m in enumerate(model_names, 1):
            st.markdown(f"**{i}.** `{m}`")
        st.caption(f"{len(model_names)} models will train in parallel across {alloc['cpu_allocated']} worker threads.")


    # ── Step 4: Training ──────────────────────────────────────────────────────
    st.markdown('<div class="section-label">Step 4 — Parallel Training</div>', unsafe_allow_html=True)

    if not _orch_ok:
        st.error("Orchestrator module is not available. See details below.")
        if _orch_err:
            with st.expander("🔍 Import Error Details (for debugging)", expanded=True):
                st.code(_orch_err, language="python")
        st.stop()

    if not st.session_state["trained"]:
        if st.button("🚀 Start Training", type="primary", use_container_width=False):
            _logs = []

            def _log(msg):
                _logs.append(msg)

            progress = st.progress(0.0, text="Initialising ...")
            log_box  = st.empty()

            def _progress(frac, label):
                progress.progress(frac, text=label)
                if _logs:
                    log_box.markdown(
                        '<div class="logwin">' + "<br>".join(_logs[-25:]) + "</div>",
                        unsafe_allow_html=True,
                    )

            try:
                with st.spinner("Training in progress ..."):
                    result = run_pipeline(
                        df=df, analysis=an,
                        model_names=model_names, allocation=alloc,
                        test_size=test_split,
                        log_fn=_log, progress_fn=_progress,
                        enable_hpo=enable_hpo,
                        enable_shap=enable_shap,
                        enable_ensemble=enable_ensemble,
                        enable_report=enable_report,
                    )
            except Exception as e:
                import traceback
                st.error(f"Pipeline error: {e}")
                st.code(traceback.format_exc())
                st.stop()

            progress.progress(1.0, text="✅ Complete")
            st.session_state["logs"]    = _logs
            st.session_state["result"]  = result
            st.session_state["trained"] = True
            st.rerun()
    else:
        st.success("✅ Training complete.")
        if st.button("🔁 Re-run training"):
            st.session_state["trained"] = False
            st.session_state["result"]  = None
            st.session_state["logs"]    = []
            st.rerun()


    # ── Step 5: Results ───────────────────────────────────────────────────────
    if st.session_state["trained"] and st.session_state["result"]:
        res        = st.session_state["result"]
        results    = res.get("results", [])
        best       = res.get("best", None)
        summary_df = res.get("summary_df", pd.DataFrame())
        speedup_info = res.get("speedup_info", {})

        st.markdown('<div class="section-label">Step 5 — Results & Visualizations</div>', unsafe_allow_html=True)

        if not results:
            st.error("No models trained successfully. Check the logs.")
        else:
            # ── Parallel Execution Time Banner ────────────────────────────────
            if speedup_info:
                wall   = speedup_info.get("wall", 0)
                sp_val = speedup_info.get("speedup", 1)
                st.markdown(f"""
                <div class="speedup-banner">
                  <div class="sp-icon">⚡</div>
                  <div>
                    <span class="sp-lbl">Parallel Execution Time</span>
                    <span class="sp-val">{wall:.1f}s</span>
                  </div>
                  <div>
                    <span class="sp-lbl">Speedup Factor</span>
                    <span class="sp-val">{sp_val:.2f}×</span>
                  </div>
                  <div style="font-size:0.82rem; color:#6b7280;">
                    {len(results)} models trained concurrently using ThreadPoolExecutor
                  </div>
                </div>
                """, unsafe_allow_html=True)

            # ── Best Model Card ────────────────────────────────────────────────
            if best:
                primary = "accuracy" if task_type == "classification" else "r2"
                plabel  = "Accuracy"  if task_type == "classification" else "R² Score"
                pval    = best.get("metrics", {}).get(primary, 0)
                cv_m    = best.get("cv_mean", 0)
                cv_s    = best.get("cv_std", 0)
                train_t = best.get("train_time", "N/A")
                ram_mb  = best.get("peak_ram_mb", 0)

                st.markdown(f"""
                <div class="best-block">
                  <div class="b-name">🏆 Best Model: {best.get('name','Unknown')}</div>
                  <div class="b-detail">
                    {plabel}: <strong>{pval:.4f}</strong> &nbsp;|&nbsp;
                    CV Score: <strong>{cv_m:.4f} ± {cv_s:.4f}</strong><br>
                    Train time: {train_t}s &nbsp;|&nbsp;
                    Peak RAM: {ram_mb:.0f} MB
                  </div>
                </div>
                """, unsafe_allow_html=True)

                # Download button
                buf = io.BytesIO()
                joblib.dump(best["model"], buf)
                buf.seek(0)
                dl1, dl2 = st.columns(2)
                with dl1:
                    st.download_button(
                        "⬇️ Download Best Model (.pkl)",
                        data=buf,
                        file_name=f"{best['name'].lower().replace(' ','_')}_model.pkl",
                        mime="application/octet-stream",
                        use_container_width=True,
                    )
                with dl2:
                    pdf_path = res.get("pdf_report_path", "")
                    if pdf_path and os.path.isfile(pdf_path):
                        with open(pdf_path, "rb") as _f:
                            st.download_button(
                                "📄 Download PDF Report",
                                data=_f.read(),
                                file_name=os.path.basename(pdf_path),
                                mime="application/pdf",
                                use_container_width=True,
                            )

            # ── Summary Table ──────────────────────────────────────────────────
            st.markdown("**📊 All Model Results**")
            if not summary_df.empty:
                st.dataframe(summary_df.style.format(precision=4), use_container_width=True)

            # ════════════════════════════════════════
            # VISUALIZATIONS — 3 clean charts
            # ════════════════════════════════════════
            st.markdown('<div class="section-label">Model Visualizations</div>', unsafe_allow_html=True)

            primary = "accuracy" if task_type == "classification" else "r2"
            plabel  = "Accuracy"  if task_type == "classification" else "R² Score"

            BG      = "#ffffff"
            BG2     = "#f7f8fa"
            GREEN   = "#15803d"
            BORDER  = "#e5e7eb"
            TXT     = "#374151"
            PALETTE = ["#16a34a","#2563eb","#d97706","#dc2626","#7c3aed","#0891b2","#059669","#ea580c"]

            def _fig(w=7, h=4.5):
                fig, ax = plt.subplots(figsize=(w, h), facecolor=BG)
                ax.set_facecolor(BG2)
                for sp in ax.spines.values():
                    sp.set_edgecolor(BORDER)
                ax.tick_params(colors=TXT, labelsize=9)
                ax.xaxis.label.set_color(TXT)
                ax.yaxis.label.set_color(TXT)
                ax.title.set_color(GREEN)
                ax.grid(True, color=BORDER, linewidth=0.5, alpha=0.8)
                return fig, ax

            if not summary_df.empty and primary in summary_df.columns:
                models_list = summary_df["Model"].tolist()
                colors_list = PALETTE[:len(models_list)]

                # ── Row 1: Performance + CV Score side by side ────────────────
                c1, c2 = st.columns(2)

                with c1:
                    fig, ax = _fig()
                    vals = summary_df[primary].tolist()
                    bars = ax.barh(models_list, vals, color=colors_list, edgecolor=BORDER, height=0.6)
                    best_idx = vals.index(max(vals))
                    bars[best_idx].set_edgecolor("#16a34a")
                    bars[best_idx].set_linewidth(2.5)
                    for bar, v in zip(bars, vals):
                        ax.text(v + 0.01, bar.get_y() + bar.get_height()/2,
                                f"{v:.3f}", va="center", color=GREEN, fontsize=9, fontweight="bold")
                    ax.set_xlabel(plabel, fontsize=9)
                    ax.set_xlim(0, 1.15)
                    ax.set_title(plabel, fontsize=12, fontweight="bold", pad=10)
                    fig.tight_layout()
                    st.pyplot(fig)
                    plt.close(fig)

                with c2:
                    if "CV Mean" in summary_df.columns:
                        fig, ax = _fig()
                        cv_means = summary_df["CV Mean"].tolist()
                        cv_stds  = summary_df["CV Std"].tolist() if "CV Std" in summary_df.columns else [0]*len(cv_means)
                        ax.bar(range(len(models_list)), cv_means, color=colors_list,
                               edgecolor=BORDER, width=0.6,
                               yerr=cv_stds, capsize=5,
                               error_kw={"ecolor": "#d97706", "elinewidth": 2})
                        ax.set_xticks(range(len(models_list)))
                        ax.set_xticklabels(models_list, rotation=25, ha="right", fontsize=8, color=TXT)
                        ax.set_ylabel("CV Score", fontsize=9)
                        ax.set_ylim(0, 1.15)
                        ax.set_title("Cross-Validation Score (mean ± std)", fontsize=12, fontweight="bold", pad=10)
                        for bar, m in zip(ax.patches, cv_means):
                            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.03,
                                    f"{m:.3f}", ha="center", va="bottom", color=GREEN, fontsize=8, fontweight="bold")
                        fig.tight_layout()
                        st.pyplot(fig)
                        plt.close(fig)

                # ── Row 2: Training Time full width ────────────────────────────
                if "Train Time (s)" in summary_df.columns:
                    fig, ax = _fig(w=11, h=4)
                    times = summary_df["Train Time (s)"].tolist()
                    bars  = ax.bar(range(len(models_list)), times, color=colors_list,
                                   edgecolor=BORDER, width=0.55)
                    ax.set_xticks(range(len(models_list)))
                    ax.set_xticklabels(models_list, rotation=20, ha="right", fontsize=9, color=TXT)
                    ax.set_ylabel("Seconds", fontsize=9)
                    ax.set_title("Training Time per Model (seconds)", fontsize=12, fontweight="bold", pad=10)
                    for bar, t in zip(bars, times):
                        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                                f"{t:.2f}s", ha="center", va="bottom", color=GREEN, fontsize=9, fontweight="bold")
                    fig.tight_layout()
                    st.pyplot(fig)
                    plt.close(fig)

            # ── SHAP images ───────────────────────────────────────────────────
            shap_imgs = [
                (r["name"], p)
                for r in results
                for p in r.get("plot_paths", [])
                if "shap_" in os.path.basename(p) and os.path.isfile(p)
            ]
            if shap_imgs:
                st.markdown('<div class="section-label">SHAP Feature Importance</div>', unsafe_allow_html=True)
                cols = st.columns(min(len(shap_imgs), 2))
                for idx, (mdl, path) in enumerate(shap_imgs):
                    with cols[idx % len(cols)]:
                        st.image(path, caption=f"SHAP — {mdl}", use_column_width=True)



        # Training log
        if st.session_state["logs"]:
            with st.expander("📋 Training Log"):
                st.markdown(
                    '<div class="logwin">' + "<br>".join(st.session_state["logs"]) + "</div>",
                    unsafe_allow_html=True,
                )

        # MLflow info
        st.markdown('<div class="section-label">MLflow Experiment Tracking</div>', unsafe_allow_html=True)
        try:
            import mlflow
            mlruns_exists = os.path.isdir(os.path.join(ROOT, "mlruns"))
            if mlruns_exists:
                st.markdown("""
                <div class="mlflow-block">
                  All training runs have been logged to MLflow under experiment
                  <code>CloudAutoML</code>.<br>
                  To view the dashboard, run in a terminal:<br>
                  <code>mlflow ui</code><br>
                  Then open: <code>http://localhost:5000</code>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.info("MLflow tracking available locally. Run `mlflow ui` to view experiments.")
        except Exception:
            st.info("MLflow UI available locally — not supported in cloud environment.")

else:
    # ── Landing state ──────────────────────────────────────────────────────────
    st.markdown("""
    <div class="landing-box">
      <div style="font-size:3.5rem; margin-bottom:20px;">📂</div>
      <div style="font-size:1.25rem; font-weight:700; color:#58d68d; margin-bottom:10px;">
        Upload a CSV file to get started
      </div>
      <div style="font-size:0.9rem; color:#8b949e; max-width:480px; margin:0 auto;">
        Supports classification and regression tasks &middot;
        Auto-detects target column &middot; No code required
      </div>
    </div>
    """, unsafe_allow_html=True)
