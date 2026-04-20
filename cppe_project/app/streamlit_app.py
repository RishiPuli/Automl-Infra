"""
app/streamlit_app.py
---------------------
CloudAutoML – Resource-Aware AutoML Platform
Dark-green premium theme | Mobile-responsive | No prediction panel.

Cloud-hardened:
  - All psutil calls wrapped in try/except with fallbacks
  - All file writes use relative paths
  - mlruns/ references wrapped in try/except (missing dir = no crash)
  - No hardcoded absolute paths anywhere
"""

import sys
import os

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

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

try:
    from core.orchestrator import run_pipeline
    _orch_ok = True
except Exception:
    _orch_ok = False

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
# CSS — Dark Premium Theme, Mobile First
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

  /* ── Global ── */
  html, body, [class*="css"] {
    font-family: 'Inter', sans-serif !important;
    background-color: #0d1117 !important;
    color: #e6edf3 !important;
  }

  /* ── Sidebar ── */
  [data-testid="stSidebar"] {
    background: #161b22 !important;
    border-right: 1px solid #30363d !important;
  }
  [data-testid="stSidebar"] .stMarkdown p,
  [data-testid="stSidebar"] label,
  [data-testid="stSidebar"] .stMarkdown {
    color: #c9d1d9 !important;
  }
  [data-testid="stSidebar"] h3 {
    color: #58d68d !important;
  }

  /* ── Hero header ── */
  .hero {
    background: linear-gradient(135deg, #1a6b3c 0%, #22863a 50%, #2ea04f 100%);
    border-radius: 16px;
    padding: 40px 36px 32px 36px;
    margin-bottom: 28px;
    color: #ffffff;
    box-shadow: 0 8px 32px rgba(46,160,79,0.25);
    border: 1px solid #2ea04f40;
  }
  .hero h1 {
    font-size: clamp(1.6rem, 4vw, 2.4rem);
    font-weight: 800;
    margin: 0 0 8px 0;
    letter-spacing: -0.5px;
  }
  .hero p {
    font-size: clamp(0.85rem, 2vw, 1rem);
    opacity: 0.9;
    margin: 0;
  }

  /* ── Section label ── */
  .section-label {
    font-size: 0.7rem;
    font-weight: 700;
    letter-spacing: 2.5px;
    text-transform: uppercase;
    color: #58d68d;
    border-left: 4px solid #2ea04f;
    padding: 4px 0 4px 14px;
    margin: 32px 0 18px 0;
    background: #1c2b1c;
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
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 12px;
    padding: 18px 16px;
    text-align: center;
    box-shadow: 0 2px 8px rgba(0,0,0,0.3);
    transition: transform 0.2s, border-color 0.2s;
  }
  .metric-card:hover {
    transform: translateY(-2px);
    border-color: #2ea04f;
  }
  .metric-card .m-val {
    font-size: clamp(1.3rem, 3vw, 1.8rem);
    font-weight: 700;
    color: #58d68d;
    line-height: 1;
  }
  .metric-card .m-lbl {
    font-size: 0.68rem;
    color: #8b949e;
    margin-top: 6px;
    letter-spacing: 0.5px;
    text-transform: uppercase;
  }

  /* ── Info tiles ── */
  .info-row { display: flex; gap: 10px; flex-wrap: wrap; margin: 14px 0; }
  .info-tile {
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 8px;
    padding: 8px 16px;
    font-size: 0.85rem;
    color: #c9d1d9;
  }
  .info-tile b { color: #58d68d; }
  .info-tile code { 
    background: #1c2b1c; 
    color: #58d68d; 
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
  .tag-clf  { background: #1c2b1c; color: #58d68d; border: 1px solid #2ea04f; }
  .tag-reg  { background: #2b2200; color: #e3b341; border: 1px solid #9e6a03; }
  .tag-sm   { background: #1a2b40; color: #79c0ff; border: 1px solid #1f6feb; }
  .tag-med  { background: #2b2200; color: #e3b341; border: 1px solid #9e6a03; }
  .tag-lg   { background: #3d1a1a; color: #f85149; border: 1px solid #da3633; }

  /* ── Allocation block ── */
  .alloc-block {
    background: #161b22;
    border: 1px solid #30363d;
    border-left: 4px solid #2ea04f;
    border-radius: 10px;
    padding: 20px 22px;
    font-size: 0.85rem;
    line-height: 2.1;
    color: #c9d1d9;
    margin-bottom: 16px;
  }
  .alloc-block .key { color: #8b949e; }
  .alloc-block .val { color: #58d68d; font-weight: 600; }

  /* ── Best model block ── */
  .best-block {
    background: linear-gradient(135deg, #1c2b1c, #0d220d);
    border: 1px solid #2ea04f;
    border-radius: 12px;
    padding: 22px 26px;
    margin-bottom: 20px;
    box-shadow: 0 4px 20px rgba(46,160,79,0.2);
  }
  .best-block .b-name {
    font-size: clamp(1rem, 3vw, 1.25rem);
    font-weight: 700;
    color: #58d68d;
  }
  .best-block .b-detail {
    font-size: 0.85rem;
    color: #c9d1d9;
    margin-top: 10px;
    line-height: 1.9;
  }
  .best-block .b-detail strong { color: #58d68d; }

  /* ── Speedup banner ── */
  .speedup-banner {
    background: linear-gradient(90deg, #1a3a4a, #0f2030);
    border: 1px solid #1f6feb;
    border-radius: 10px;
    padding: 16px 22px;
    color: #79c0ff;
    font-size: 0.9rem;
    margin: 16px 0;
    display: flex;
    align-items: center;
    gap: 14px;
    flex-wrap: wrap;
  }
  .speedup-banner .sp-icon { font-size: 1.5rem; }
  .speedup-banner .sp-val  { font-size: 1.3rem; font-weight: 700; color: #58d68d; }
  .speedup-banner .sp-lbl  { font-size: 0.78rem; color: #8b949e; display: block; }

  /* ── Viz section ── */
  .viz-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
    gap: 16px;
    margin: 18px 0;
  }
  .viz-card {
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 10px;
    padding: 16px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.3);
  }
  .viz-card-title {
    font-size: 0.78rem;
    font-weight: 600;
    color: #8b949e;
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-bottom: 10px;
  }

  /* ── Log window ── */
  .logwin {
    background: #0d1117;
    border: 1px solid #30363d;
    border-radius: 8px;
    padding: 14px 16px;
    font-family: 'Courier New', monospace;
    font-size: 0.76rem;
    color: #7ee787;
    max-height: 220px;
    overflow-y: auto;
    white-space: pre-wrap;
    line-height: 1.7;
  }

  /* ── Notice ── */
  .notice {
    background: #2b2200;
    border: 1px solid #9e6a03;
    border-radius: 8px;
    padding: 10px 16px;
    font-size: 0.84rem;
    color: #e3b341;
    margin: 8px 0;
  }

  /* ── MLflow block ── */
  .mlflow-block {
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 10px;
    padding: 18px 22px;
    font-size: 0.84rem;
    color: #c9d1d9;
    line-height: 2;
  }
  .mlflow-block code {
    background: #1c2b1c;
    color: #58d68d;
    padding: 2px 8px;
    border-radius: 4px;
    font-family: 'Courier New', monospace;
  }

  /* ── Landing box ── */
  .landing-box {
    background: #161b22;
    border: 2px dashed #30363d;
    border-radius: 16px;
    padding: 60px 40px;
    text-align: center;
    color: #8b949e;
    margin-top: 16px;
  }

  /* ── Streamlit overrides ── */
  .stDataFrame { border: 1px solid #30363d !important; border-radius: 8px; }
  .stButton > button {
    background: #238636 !important;
    color: #ffffff !important;
    border: 1px solid #2ea04f !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    padding: 0.5rem 1.5rem !important;
    transition: all 0.2s;
    font-family: 'Inter', sans-serif !important;
  }
  .stButton > button:hover {
    background: #2ea04f !important;
    border-color: #3fb950 !important;
    box-shadow: 0 0 12px rgba(46,160,79,0.3) !important;
    transform: translateY(-1px);
  }
  .stDownloadButton > button {
    background: #161b22 !important;
    color: #58d68d !important;
    border: 1px solid #2ea04f !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
  }
  .stTabs [data-baseweb="tab"] {
    color: #8b949e !important;
    font-weight: 500;
  }
  .stTabs [aria-selected="true"] {
    color: #58d68d !important;
    border-bottom: 2px solid #58d68d !important;
  }
  .stProgress > div > div { background-color: #2ea04f !important; }
  .stSlider [data-testid="stThumbValue"] { color: #58d68d !important; }
  div[data-testid="stExpander"] {
    background: #161b22 !important;
    border: 1px solid #30363d !important;
    border-radius: 8px !important;
  }
  div[data-testid="stExpander"] summary {
    color: #c9d1d9 !important;
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
        st.error("Orchestrator module is not available. Check your installation.")
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
                seq    = speedup_info.get("sequential_estimate", 0)
                sp_val = speedup_info.get("speedup", 1)
                st.markdown(f"""
                <div class="speedup-banner">
                  <div class="sp-icon">⚡</div>
                  <div>
                    <span class="sp-lbl">Parallel Wall Time</span>
                    <span class="sp-val">{wall:.1f}s</span>
                  </div>
                  <div>
                    <span class="sp-lbl">Sequential Estimate</span>
                    <span class="sp-val">{seq:.1f}s</span>
                  </div>
                  <div>
                    <span class="sp-lbl">Parallel Speedup</span>
                    <span class="sp-val">{sp_val:.2f}×</span>
                  </div>
                  <div style="font-size:0.82rem; color:#8b949e;">
                    Trained {len(results)} models concurrently using ThreadPoolExecutor
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

            # ═══════════════════════════════════════════════════════════════════
            # VISUALIZATIONS
            # ═══════════════════════════════════════════════════════════════════
            st.markdown('<div class="section-label">Model Visualizations</div>', unsafe_allow_html=True)

            primary = "accuracy" if task_type == "classification" else "r2"
            plabel  = "Accuracy"  if task_type == "classification" else "R² Score"

            # Dark plot style
            DARK_BG    = "#0d1117"
            CARD_BG    = "#161b22"
            GREEN      = "#2ea04f"
            GREEN_L    = "#58d68d"
            BORDER     = "#30363d"
            TEXT_COLOR = "#c9d1d9"
            SUBTEXT    = "#8b949e"
            PALETTE    = ["#2ea04f","#1f6feb","#e3b341","#f85149","#bc8cff","#79c0ff","#58d68d","#ffa657"]

            def _dark_fig(w=7, h=4):
                fig, ax = plt.subplots(figsize=(w, h), facecolor=DARK_BG)
                ax.set_facecolor(CARD_BG)
                for spine in ax.spines.values():
                    spine.set_edgecolor(BORDER)
                ax.tick_params(colors=TEXT_COLOR, labelsize=8)
                ax.xaxis.label.set_color(TEXT_COLOR)
                ax.yaxis.label.set_color(TEXT_COLOR)
                ax.title.set_color(GREEN_L)
                ax.grid(True, color=BORDER, linewidth=0.5, alpha=0.6)
                return fig, ax

            # Build display data from summary_df
            if not summary_df.empty and primary in summary_df.columns:

                tab_perf, tab_time, tab_cv, tab_radar, tab_shap = st.tabs([
                    "📈 Performance", "⏱️ Training Time", "🔄 Cross-Validation", "🕸️ Radar Chart", "🔍 SHAP"
                ])

                models_list = summary_df["Model"].tolist()
                colors_list = PALETTE[:len(models_list)]

                # ── Tab 1: Performance Bar Chart ──────────────────────────────
                with tab_perf:
                    col_v1, col_v2 = st.columns([1, 1])

                    with col_v1:
                        fig, ax = _dark_fig(6, 4)
                        vals = summary_df[primary].tolist()
                        bars = ax.barh(models_list, vals, color=colors_list, edgecolor=BORDER, height=0.6)
                        ax.set_xlabel(plabel, color=TEXT_COLOR, fontsize=9)
                        ax.set_title(f"Model {plabel} Comparison", fontsize=10, color=GREEN_L, fontweight="bold")
                        ax.set_xlim(0, 1.05)
                        for bar, val in zip(bars, vals):
                            ax.text(val + 0.01, bar.get_y() + bar.get_height()/2,
                                    f"{val:.3f}", va="center", color=GREEN_L, fontsize=8, fontweight="bold")
                        # Highlight best
                        best_idx = vals.index(max(vals))
                        bars[best_idx].set_edgecolor("#58d68d")
                        bars[best_idx].set_linewidth(2)
                        fig.tight_layout()
                        st.pyplot(fig)
                        plt.close(fig)

                    with col_v2:
                        # F1 / MAE secondary metric
                        sec_col = "f1_score" if "f1_score" in summary_df.columns else ("mae" if "mae" in summary_df.columns else None)
                        if sec_col:
                            sec_label = "F1 Score" if sec_col == "f1_score" else "MAE"
                            fig, ax = _dark_fig(6, 4)
                            vals2 = summary_df[sec_col].tolist()
                            ax.bar(models_list, vals2, color=colors_list, edgecolor=BORDER, width=0.6)
                            ax.set_ylabel(sec_label, color=TEXT_COLOR, fontsize=9)
                            ax.set_title(f"{sec_label} by Model", fontsize=10, color=GREEN_L, fontweight="bold")
                            ax.set_xticks(range(len(models_list)))
                            ax.set_xticklabels(models_list, rotation=25, ha="right", fontsize=8, color=TEXT_COLOR)
                            fig.tight_layout()
                            st.pyplot(fig)
                            plt.close(fig)
                        else:
                            # RMSE fallback
                            if "rmse" in summary_df.columns:
                                fig, ax = _dark_fig(6, 4)
                                vals_rmse = summary_df["rmse"].tolist()
                                ax.bar(models_list, vals_rmse, color=colors_list, edgecolor=BORDER, width=0.6)
                                ax.set_ylabel("RMSE", color=TEXT_COLOR, fontsize=9)
                                ax.set_title("RMSE by Model", fontsize=10, color=GREEN_L, fontweight="bold")
                                ax.set_xticks(range(len(models_list)))
                                ax.set_xticklabels(models_list, rotation=25, ha="right", fontsize=8, color=TEXT_COLOR)
                                fig.tight_layout()
                                st.pyplot(fig)
                                plt.close(fig)

                # ── Tab 2: Training Time + RAM ────────────────────────────────
                with tab_time:
                    col_t1, col_t2 = st.columns([1, 1])

                    with col_t1:
                        if "Train Time (s)" in summary_df.columns:
                            fig, ax = _dark_fig(6, 4)
                            times = summary_df["Train Time (s)"].tolist()
                            bars = ax.bar(models_list, times, color=colors_list, edgecolor=BORDER, width=0.6)
                            ax.set_ylabel("Train Time (s)", color=TEXT_COLOR, fontsize=9)
                            ax.set_title("Training Time per Model", fontsize=10, color=GREEN_L, fontweight="bold")
                            ax.set_xticks(range(len(models_list)))
                            ax.set_xticklabels(models_list, rotation=25, ha="right", fontsize=8, color=TEXT_COLOR)
                            for bar, t in zip(bars, times):
                                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                                        f"{t:.2f}s", ha="center", va="bottom", color=GREEN_L, fontsize=8)
                            fig.tight_layout()
                            st.pyplot(fig)
                            plt.close(fig)

                    with col_t2:
                        if "Peak RAM (MB)" in summary_df.columns:
                            fig, ax = _dark_fig(6, 4)
                            rams = summary_df["Peak RAM (MB)"].tolist()
                            ax.barh(models_list, rams, color="#1f6feb", edgecolor=BORDER, height=0.6)
                            ax.set_xlabel("Peak RAM (MB)", color=TEXT_COLOR, fontsize=9)
                            ax.set_title("Peak Memory Usage", fontsize=10, color="#79c0ff", fontweight="bold")
                            ax.tick_params(colors=TEXT_COLOR, labelsize=8)
                            for spine in ax.spines.values():
                                spine.set_edgecolor(BORDER)
                            fig.tight_layout()
                            st.pyplot(fig)
                            plt.close(fig)

                    # Parallel vs Sequential time comparison
                    if speedup_info:
                        st.markdown("---")
                        col_sp1, col_sp2, col_sp3 = st.columns(3)
                        with col_sp1:
                            wall_t = speedup_info.get("wall", 0)
                            st.metric("⚡ Parallel Wall Time", f"{wall_t:.1f}s")
                        with col_sp2:
                            seq_t = speedup_info.get("sequential_estimate", 0)
                            st.metric("🐌 Sequential Estimate", f"{seq_t:.1f}s")
                        with col_sp3:
                            sp = speedup_info.get("speedup", 1)
                            st.metric("🚀 Speedup Factor", f"{sp:.2f}×", delta=f"saved {max(0, seq_t - wall_t):.1f}s")

                        # Speedup visual
                        fig, ax = _dark_fig(8, 3)
                        categories = ["Sequential\n(estimate)", "Parallel\n(actual)"]
                        values     = [seq_t, wall_t]
                        bar_colors = ["#f85149", "#2ea04f"]
                        bars = ax.bar(categories, values, color=bar_colors, edgecolor=BORDER, width=0.4)
                        for bar, v in zip(bars, values):
                            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
                                    f"{v:.1f}s", ha="center", va="bottom", color=TEXT_COLOR, fontsize=10, fontweight="bold")
                        ax.set_ylabel("Time (seconds)", color=TEXT_COLOR, fontsize=9)
                        ax.set_title(f"Parallel Execution Speedup: {sp:.2f}×", fontsize=11, color=GREEN_L, fontweight="bold")
                        ax.set_facecolor(CARD_BG)
                        fig.patch.set_facecolor(DARK_BG)
                        fig.tight_layout()
                        st.pyplot(fig)
                        plt.close(fig)

                # ── Tab 3: Cross-Validation ───────────────────────────────────
                with tab_cv:
                    if "CV Mean" in summary_df.columns and "CV Std" in summary_df.columns:
                        fig, ax = _dark_fig(8, 4.5)
                        cv_means = summary_df["CV Mean"].tolist()
                        cv_stds  = summary_df["CV Std"].tolist()
                        x = range(len(models_list))
                        bars = ax.bar(x, cv_means, color=colors_list, edgecolor=BORDER, width=0.55,
                                      yerr=cv_stds, capsize=6, error_kw={"ecolor": "#e3b341", "elinewidth": 2})
                        ax.set_xticks(list(x))
                        ax.set_xticklabels(models_list, rotation=25, ha="right", fontsize=8, color=TEXT_COLOR)
                        ax.set_ylabel("CV Score", color=TEXT_COLOR, fontsize=9)
                        ax.set_title("Cross-Validation Scores (Mean ± Std)", fontsize=11, color=GREEN_L, fontweight="bold")
                        ax.set_ylim(0, 1.1)
                        for bar, m, s in zip(bars, cv_means, cv_stds):
                            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + s + 0.02,
                                    f"{m:.3f}", ha="center", va="bottom", color=GREEN_L, fontsize=8, fontweight="bold")
                        fig.tight_layout()
                        st.pyplot(fig)
                        plt.close(fig)

                        # Heatmap of all metrics
                        metric_cols = [c for c in summary_df.columns
                                       if c not in ["Model", "Workers"] and summary_df[c].dtype != object]
                        if len(metric_cols) >= 2:
                            st.markdown("**📋 Metrics Heatmap**")
                            heat_df = summary_df[["Model"] + metric_cols].set_index("Model")
                            norm_df = (heat_df - heat_df.min()) / (heat_df.max() - heat_df.min() + 1e-9)
                            fig2, ax2 = plt.subplots(
                                figsize=(min(12, 2 + len(metric_cols)*1.5), max(3, len(models_list)*0.7)),
                                facecolor=DARK_BG
                            )
                            ax2.set_facecolor(DARK_BG)
                            im = ax2.imshow(norm_df.values, cmap="YlGn", aspect="auto", vmin=0, vmax=1)
                            ax2.set_xticks(range(len(metric_cols)))
                            ax2.set_xticklabels(metric_cols, rotation=30, ha="right", color=TEXT_COLOR, fontsize=8)
                            ax2.set_yticks(range(len(models_list)))
                            ax2.set_yticklabels(models_list, color=TEXT_COLOR, fontsize=8)
                            for i in range(len(models_list)):
                                for j in range(len(metric_cols)):
                                    ax2.text(j, i, f"{heat_df.values[i,j]:.3f}",
                                             ha="center", va="center", fontsize=7, color="#0d1117", fontweight="bold")
                            ax2.set_title("Normalized Metrics Heatmap (higher = better)", color=GREEN_L, fontsize=10, fontweight="bold")
                            cbar = fig2.colorbar(im, ax=ax2)
                            cbar.ax.tick_params(colors=TEXT_COLOR, labelsize=7)
                            fig2.tight_layout()
                            st.pyplot(fig2)
                            plt.close(fig2)

                # ── Tab 4: Radar Chart ────────────────────────────────────────
                with tab_radar:
                    radar_metrics = [c for c in [primary, "CV Mean", "Train Time (s)", "Peak RAM (MB)"]
                                     if c in summary_df.columns]
                    if len(radar_metrics) >= 3 and len(models_list) >= 2:
                        radar_data = summary_df[["Model"] + radar_metrics].copy()
                        # Normalize each metric 0–1 (invert time/RAM so higher = better)
                        for col in radar_metrics:
                            mn, mx = radar_data[col].min(), radar_data[col].max()
                            if mx > mn:
                                if col in ["Train Time (s)", "Peak RAM (MB)"]:
                                    radar_data[col] = 1 - (radar_data[col] - mn) / (mx - mn)
                                else:
                                    radar_data[col] = (radar_data[col] - mn) / (mx - mn)
                            else:
                                radar_data[col] = 1.0

                        N      = len(radar_metrics)
                        angles = [n / float(N) * 2 * np.pi for n in range(N)]
                        angles += angles[:1]

                        fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True), facecolor=DARK_BG)
                        ax.set_facecolor(CARD_BG)
                        ax.spines["polar"].set_color(BORDER)
                        ax.tick_params(colors=TEXT_COLOR, labelsize=9)
                        ax.set_xticks(angles[:-1])
                        ax.set_xticklabels(radar_metrics, color=TEXT_COLOR, fontsize=9)
                        ax.set_yticklabels([])
                        ax.yaxis.grid(True, color=BORDER, linewidth=0.5)
                        ax.xaxis.grid(True, color=BORDER, linewidth=0.5)

                        for idx, row in radar_data.iterrows():
                            vals_r = row[radar_metrics].tolist()
                            vals_r += vals_r[:1]
                            color = colors_list[idx % len(colors_list)]
                            ax.plot(angles, vals_r, linewidth=2, color=color)
                            ax.fill(angles, vals_r, alpha=0.15, color=color)
                        ax.set_title("Model Capability Radar", color=GREEN_L, fontsize=12,
                                     fontweight="bold", pad=20)
                        handles = [mpatches.Patch(color=colors_list[i], label=m)
                                   for i, m in enumerate(models_list)]
                        ax.legend(handles=handles, loc="upper right", bbox_to_anchor=(1.35, 1.15),
                                  facecolor=CARD_BG, edgecolor=BORDER, fontsize=8,
                                  labelcolor=TEXT_COLOR)
                        fig.tight_layout()
                        st.pyplot(fig)
                        plt.close(fig)
                    else:
                        st.info("Need at least 2 models and 3 metrics for radar chart.")

                # ── Tab 5: SHAP ───────────────────────────────────────────────
                with tab_shap:
                    shap_imgs = [
                        (r["name"], p)
                        for r in results
                        for p in r.get("plot_paths", [])
                        if "shap_" in os.path.basename(p) and os.path.isfile(p)
                    ]
                    if shap_imgs:
                        for i in range(0, len(shap_imgs), 2):
                            cols = st.columns(min(2, len(shap_imgs) - i))
                            for j, col in enumerate(cols):
                                if i + j < len(shap_imgs):
                                    mdl, path = shap_imgs[i + j]
                                    with col:
                                        st.image(path, caption=f"SHAP — {mdl}", use_container_width=True)
                    else:
                        st.info("SHAP plots will appear here after training with SHAP enabled.")

            # ── Eval plots from disk ───────────────────────────────────────────
            eval_plots = [
                (r["name"], p)
                for r in results
                for p in r.get("plot_paths", [])
                if "shap_" not in os.path.basename(p) and os.path.isfile(p)
            ]
            if eval_plots:
                st.markdown('<div class="section-label">Evaluation Plots</div>', unsafe_allow_html=True)
                for i in range(0, len(eval_plots), 2):
                    cols = st.columns(min(2, len(eval_plots) - i))
                    for j, col in enumerate(cols):
                        if i + j < len(eval_plots):
                            mdl, path = eval_plots[i + j]
                            with col:
                                st.image(path, caption=mdl, use_container_width=True)

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
