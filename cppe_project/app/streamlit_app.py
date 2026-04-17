"""
app/streamlit_app.py
---------------------
CloudAutoML – Resource-Aware AutoML Platform
White + Light-Green theme  |  Predict panel at the bottom.

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
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# CSS  —  White + Light-Green theme
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

  /* ── Global ── */
  html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    background-color: #f5faf5;
    color: #1a2e1a;
  }

  /* ── Sidebar ── */
  [data-testid="stSidebar"] {
    background: #ffffff;
    border-right: 1px solid #d6ead6;
  }
  [data-testid="stSidebar"] .stMarkdown p,
  [data-testid="stSidebar"] label {
    color: #2d4a2d;
  }

  /* ── Hero header ── */
  .hero {
    background: linear-gradient(135deg, #1e7a3e 0%, #28a85e 50%, #42c97a 100%);
    border-radius: 14px;
    padding: 36px 40px 30px 40px;
    margin-bottom: 32px;
    color: #ffffff;
    box-shadow: 0 4px 24px rgba(30,122,62,0.18);
  }
  .hero h1 {
    font-size: 2.2rem;
    font-weight: 700;
    margin: 0 0 6px 0;
    letter-spacing: -0.5px;
  }
  .hero p {
    font-size: 1rem;
    opacity: 0.88;
    margin: 0;
  }

  /* ── Section headers ── */
  .section-label {
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: #1e7a3e;
    border-left: 4px solid #28a85e;
    padding: 3px 0 3px 12px;
    margin: 32px 0 18px 0;
    background: #eafaf0;
    border-radius: 0 6px 6px 0;
  }

  /* ── Metric cards ── */
  .metric-grid {
    display: flex;
    gap: 14px;
    flex-wrap: wrap;
    margin-bottom: 22px;
  }
  .metric-card {
    background: #ffffff;
    border: 1.5px solid #c8e6c8;
    border-radius: 10px;
    padding: 16px 22px;
    min-width: 120px;
    flex: 1;
    box-shadow: 0 2px 8px rgba(30,122,62,0.07);
  }
  .metric-card .m-val {
    font-size: 1.65rem;
    font-weight: 700;
    color: #1e7a3e;
    line-height: 1;
  }
  .metric-card .m-lbl {
    font-size: 0.72rem;
    color: #6a8f6a;
    margin-top: 6px;
    letter-spacing: 0.5px;
    text-transform: uppercase;
  }

  /* ── Info tiles ── */
  .info-row { display: flex; gap: 10px; flex-wrap: wrap; margin: 14px 0; }
  .info-tile {
    background: #f0faf0;
    border: 1px solid #c8e6c8;
    border-radius: 6px;
    padding: 8px 18px;
    font-size: 0.84rem;
    color: #2d4a2d;
  }
  .info-tile b { color: #1e7a3e; }

  /* ── Tags ── */
  .tag {
    display: inline-block;
    border-radius: 4px;
    padding: 3px 12px;
    font-size: 0.74rem;
    font-weight: 600;
    letter-spacing: 0.5px;
  }
  .tag-clf  { background: #e6f9ee; color: #1e7a3e; border: 1px solid #28a85e; }
  .tag-reg  { background: #fff8e1; color: #b8860b; border: 1px solid #f0c040; }
  .tag-sm   { background: #e3f2fd; color: #1565c0; border: 1px solid #64b5f6; }
  .tag-med  { background: #fff8e1; color: #b8860b; border: 1px solid #f0c040; }
  .tag-lg   { background: #fce4ec; color: #c62828; border: 1px solid #ef9a9a; }

  /* ── Allocation block ── */
  .alloc-block {
    background: #ffffff;
    border: 1.5px solid #c8e6c8;
    border-left: 4px solid #28a85e;
    border-radius: 8px;
    padding: 18px 22px;
    font-size: 0.85rem;
    line-height: 2;
    color: #2d4a2d;
    margin-bottom: 16px;
    box-shadow: 0 2px 8px rgba(30,122,62,0.06);
  }
  .alloc-block .key { color: #6a8f6a; }
  .alloc-block .val { color: #1e7a3e; font-weight: 600; }

  /* ── Best model block ── */
  .best-block {
    background: linear-gradient(135deg, #e6f9ee, #f0faf0);
    border: 1.5px solid #28a85e;
    border-radius: 10px;
    padding: 20px 24px;
    margin-bottom: 20px;
    box-shadow: 0 2px 12px rgba(30,122,62,0.12);
  }
  .best-block .b-name {
    font-size: 1.2rem;
    font-weight: 700;
    color: #1e7a3e;
  }
  .best-block .b-detail {
    font-size: 0.84rem;
    color: #3a6b3a;
    margin-top: 8px;
    line-height: 1.8;
  }

  /* ── Log window ── */
  .logwin {
    background: #f8fff8;
    border: 1px solid #c8e6c8;
    border-radius: 6px;
    padding: 12px 16px;
    font-family: 'Courier New', monospace;
    font-size: 0.78rem;
    color: #2d4a2d;
    max-height: 220px;
    overflow-y: auto;
    white-space: pre-wrap;
    line-height: 1.6;
  }

  /* ── Notice ── */
  .notice {
    background: #fffde7;
    border: 1px solid #f9a825;
    border-radius: 6px;
    padding: 10px 16px;
    font-size: 0.84rem;
    color: #7b5800;
    margin: 8px 0;
  }

  /* ── MLflow block ── */
  .mlflow-block {
    background: #f0faf0;
    border: 1px solid #c8e6c8;
    border-radius: 8px;
    padding: 16px 20px;
    font-size: 0.84rem;
    color: #2d4a2d;
    line-height: 1.9;
  }
  .mlflow-block code {
    background: #e6f9ee;
    color: #1e7a3e;
    padding: 2px 8px;
    border-radius: 4px;
    font-family: 'Courier New', monospace;
  }

  /* ── Predict panel ── */
  .predict-hero {
    background: linear-gradient(135deg, #28a85e 0%, #42c97a 100%);
    border-radius: 12px;
    padding: 28px 36px 24px 36px;
    margin-bottom: 24px;
    color: #ffffff;
  }
  .predict-hero h2 {
    font-size: 1.5rem; font-weight: 700; margin: 0 0 4px 0;
  }
  .predict-hero p {
    margin: 0; opacity: 0.88; font-size: 0.92rem;
  }

  .predict-box {
    background: #ffffff;
    border: 1.5px solid #c8e6c8;
    border-radius: 10px;
    padding: 24px 28px;
    margin-bottom: 18px;
    box-shadow: 0 2px 10px rgba(30,122,62,0.08);
  }

  .pred-result {
    background: linear-gradient(135deg, #e6f9ee, #f0faf0);
    border: 2px solid #28a85e;
    border-radius: 10px;
    padding: 20px 28px;
    text-align: center;
    margin-top: 16px;
  }
  .pred-result .pred-label { font-size: 0.8rem; color: #6a8f6a; text-transform: uppercase; letter-spacing: 1px; }
  .pred-result .pred-value { font-size: 2rem; font-weight: 700; color: #1e7a3e; margin-top: 4px; }
  .pred-result .pred-conf  { font-size: 0.85rem; color: #3a6b3a; margin-top: 6px; }

  /* ── Streamlit overrides ── */
  .stDataFrame       { border: 1.5px solid #c8e6c8 !important; border-radius: 8px; }
  .stButton > button {
    background: #1e7a3e !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    padding: 0.5rem 1.4rem !important;
    transition: background 0.2s;
  }
  .stButton > button:hover { background: #28a85e !important; }
  #MainMenu { visibility: hidden; }
  footer    { visibility: hidden; }
  header    { visibility: hidden; }
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
    st.markdown("**📈 MLflow**")
    st.markdown("""Run: `mlflow ui` → `http://localhost:5000`""")

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
        f'<div class="info-tile"><b>Target:</b> <code style="color:#1e7a3e">{an["target_column"]}</code></div>'
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

        st.markdown('<div class="section-label">Step 5 — Results</div>', unsafe_allow_html=True)

        if not results:
            st.error("No models trained successfully. Check the logs.")
        else:
            if best:
                primary = "accuracy" if task_type == "classification" else "r2"
                plabel  = "Accuracy"  if task_type == "classification" else "R² Score"
                pval    = best.get("metrics", {}).get(primary, 0)
                cv_m    = best.get("cv_mean", 0)
                cv_s    = best.get("cv_std", 0)

                train_t  = best.get("train_time", "N/A")
                ram_mb   = best.get("peak_ram_mb", 0)
                workers  = best.get("n_jobs", 1)

                st.markdown(f"""
                <div class="best-block">
                  <div class="b-name">🏆 Best Model: {best.get('name','Unknown')}</div>
                  <div class="b-detail">
                    {plabel}: <strong>{pval:.4f}</strong> &nbsp;|&nbsp;
                    CV: <strong>{cv_m:.4f} ± {cv_s:.4f}</strong><br>
                    Train time: {train_t} s &nbsp;|&nbsp;
                    Peak RAM: {ram_mb:.0f} MB &nbsp;|&nbsp;
                    Workers: {workers}
                  </div>
                </div>
                """, unsafe_allow_html=True)

                # Download buttons
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

            # Summary table
            st.markdown("**📊 All Model Results**")
            if not summary_df.empty:
                st.dataframe(summary_df, use_container_width=True)

            # Bar charts
            primary = "accuracy" if task_type == "classification" else "r2"
            if not summary_df.empty and primary in summary_df.columns:
                c1, c2 = st.columns(2)
                with c1:
                    st.markdown("**Metric Comparison**")
                    st.bar_chart(summary_df.set_index("Model")[[primary]])
                with c2:
                    if "CV Mean" in summary_df.columns:
                        st.markdown("**Cross-Validation Mean**")
                        st.bar_chart(summary_df.set_index("Model")[["CV Mean"]])

            # SHAP plots
            shap_imgs = [
                (r["name"], p)
                for r in results
                for p in r.get("plot_paths", [])
                if "shap_" in os.path.basename(p) and os.path.isfile(p)
            ]
            if shap_imgs:
                st.markdown("**🔍 SHAP Feature Importance**")
                shap_cols = st.columns(min(len(shap_imgs), 2))
                for idx, (mdl, path) in enumerate(shap_imgs):
                    with shap_cols[idx % 2]:
                        st.image(path, caption=f"SHAP — {mdl}", use_container_width=True)

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


        # ── Step 6: Predict ───────────────────────────────────────────────────
        st.markdown("---")
        st.markdown("""
        <div class="predict-hero">
          <h2>🎯 Make Predictions</h2>
          <p>Use the trained best model to predict on new data — enter values manually or upload a CSV.</p>
        </div>
        """, unsafe_allow_html=True)

        if best is None:
            st.warning("No model available. Complete training first.")
        else:
            pred_model = best["model"]
            prep       = res.get("prep", None)
            le         = res.get("label_enc", None)

            # The exact columns the preprocessor was fit on (order matters!)
            num_c      = an.get("numerical_cols", [])
            cat_c      = an.get("categorical_cols", [])
            feat_cols  = num_c + cat_c   # same order as build_preprocessor()

            # Fallback: load from disk if prep is missing
            if prep is None:
                prep_path = os.path.join(ROOT, "models", "preprocessor.pkl")
                if os.path.isfile(prep_path):
                    try:
                        prep = joblib.load(prep_path)
                    except Exception:
                        prep = None

            # ── Feature column lists (must be defined before _apply_prep) ────
            num_c     = an.get("numerical_cols", [])
            cat_c     = an.get("categorical_cols", [])
            feat_cols = num_c + cat_c   # exact order as build_preprocessor()

            # ── Model feature-count guard ──────────────────────────────────
            expected_n = getattr(pred_model, "n_features_in_", None)

            def _apply_prep(raw_df: pd.DataFrame):
                """
                1. Select + reorder columns to match training order.
                2. Fill numeric NaN with 0.
                3. Run through the ColumnTransformer.
                Raises ValueError with a human-readable message on any mismatch.
                """
                missing_cols = [c for c in feat_cols if c not in raw_df.columns]
                if missing_cols:
                    raise ValueError(
                        f"Input is missing required columns: {missing_cols}.\n"
                        f"Required columns: {feat_cols}"
                    )
                aligned        = raw_df[feat_cols].copy()
                aligned[num_c] = aligned[num_c].fillna(0)

                if prep is not None:
                    X = prep.transform(aligned)
                else:
                    X = aligned[num_c].values if num_c else aligned.values

                # Validate output shape against what the model was trained on
                if expected_n is not None and X.shape[1] != expected_n:
                    raise ValueError(
                        f"Preprocessor output has {X.shape[1]} features but the "
                        f"current model expects {expected_n}.\n"
                        f"This usually means the model was trained on a different dataset. "
                        f"Please click '🔁 Re-run training' above to retrain on this dataset."
                    )
                return X

            tab_manual, tab_csv = st.tabs(["✏️  Manual Input", "📂  Upload CSV"])

            # ── Tab 1: Manual Input ───────────────────────────────────────────
            with tab_manual:
                st.markdown('<div class="predict-box">', unsafe_allow_html=True)
                st.markdown("**Enter feature values below:**")

                if not feat_cols:
                    st.warning("No feature columns detected. Make sure training completed successfully.")
                else:


                    # Lay out features in 3-column grid
                    input_vals = {}
                    cols_per_row = 3
                    all_feats = feat_cols
                    rows = [all_feats[i:i+cols_per_row] for i in range(0, len(all_feats), cols_per_row)]

                    for row_feats in rows:
                        cols = st.columns(len(row_feats))
                        for col, feat in zip(cols, row_feats):
                            with col:
                                if feat in cat_c:
                                    # Show unique values from the training data as selectbox
                                    uniq = df[feat].dropna().unique().tolist()[:20]
                                    input_vals[feat] = st.selectbox(
                                        feat, options=uniq, key=f"pred_{feat}"
                                    )
                                else:
                                    col_median = float(df[feat].median()) if feat in df.columns else 0.0
                                    input_vals[feat] = st.number_input(
                                        feat, value=col_median, key=f"pred_{feat}"
                                    )

                    st.markdown("</div>", unsafe_allow_html=True)

                    if st.button("🔮 Predict", key="manual_predict", use_container_width=False):
                        try:
                            row_df   = pd.DataFrame([input_vals])
                            X        = _apply_prep(row_df)
                            raw_pred = pred_model.predict(X)

                            # Decode label
                            if le is not None:
                                try:
                                    pred_label = str(le.inverse_transform(raw_pred.astype(int))[0])
                                except Exception:
                                    pred_label = str(raw_pred[0])
                            else:
                                pred_label = f"{raw_pred[0]:.4f}"

                            # Confidence
                            conf_html = ""
                            if hasattr(pred_model, "predict_proba"):
                                proba = pred_model.predict_proba(X)[0]
                                conf  = float(proba.max()) * 100
                                conf_html = f'<div class="pred-conf">Confidence: <strong>{conf:.1f}%</strong></div>'

                            pred_type = "Predicted Class" if task_type == "classification" else "Predicted Value"
                            st.markdown(f"""
                            <div class="pred-result">
                              <div class="pred-label">{pred_type}</div>
                              <div class="pred-value">{pred_label}</div>
                              {conf_html}
                            </div>
                            """, unsafe_allow_html=True)

                        except ValueError as e:
                            st.error(f"Feature mismatch: {e}")
                        except Exception as e:
                            st.error(f"Prediction failed: {e}")

            # ── Tab 2: CSV Upload ─────────────────────────────────────────────
            with tab_csv:
                st.markdown('<div class="predict-box">', unsafe_allow_html=True)
                st.markdown(
                    "Upload a CSV with the **same feature columns** used during training "
                    f"(no target column needed). Required columns: `{', '.join(feat_cols[:6])}{'...' if len(feat_cols)>6 else ''}`"
                )

                pred_csv = st.file_uploader(
                    "Upload prediction CSV", type=["csv"],
                    key="pred_csv_upload", label_visibility="collapsed",
                )

                if pred_csv is not None:
                    try:
                        pred_df = pd.read_csv(pred_csv)

                        # Drop target if accidentally included
                        tgt = an["target_column"]
                        if tgt in pred_df.columns:
                            pred_df = pred_df.drop(columns=[tgt])

                        st.markdown(f"**Preview** ({len(pred_df)} rows):")
                        st.dataframe(pred_df.head(5), use_container_width=True)

                        if st.button("🔮 Predict All Rows", key="csv_predict"):
                            try:
                                X_all     = _apply_prep(pred_df)
                                raw_preds = pred_model.predict(X_all)

                                if le is not None:
                                    try:
                                        decoded = le.inverse_transform(raw_preds.astype(int))
                                    except Exception:
                                        decoded = raw_preds
                                else:
                                    decoded = raw_preds

                                out_df = pred_df.copy()
                                out_df["Prediction"] = decoded

                                # Probabilities
                                if hasattr(pred_model, "predict_proba"):
                                    proba_all = pred_model.predict_proba(X_all)
                                    out_df["Confidence (%)"] = (proba_all.max(axis=1) * 100).round(1)

                                st.success(f"✅ {len(raw_preds)} predictions generated.")
                                st.dataframe(out_df, use_container_width=True)

                                # Download predictions
                                csv_out = out_df.to_csv(index=False).encode("utf-8")
                                st.download_button(
                                    "⬇️ Download Predictions CSV",
                                    data=csv_out,
                                    file_name="predictions.csv",
                                    mime="text/csv",
                                    use_container_width=True,
                                )

                            except ValueError as e:
                                st.error(f"Feature mismatch: {e}")
                            except Exception as e:
                                st.error(f"Batch prediction failed: {e}")

                    except Exception as e:
                        st.error(f"Could not read file: {e}")

                st.markdown("</div>", unsafe_allow_html=True)

else:
    # ── Landing state ─────────────────────────────────────────────────────────
    st.markdown("""
    <div style="
      background: #ffffff;
      border: 2px dashed #c8e6c8;
      border-radius: 14px;
      padding: 48px 40px;
      text-align: center;
      color: #6a8f6a;
    ">
      <div style="font-size:3rem; margin-bottom:16px;">📂</div>
      <div style="font-size:1.2rem; font-weight:600; color:#1e7a3e; margin-bottom:8px;">
        Upload a CSV file to get started
      </div>
      <div style="font-size:0.9rem;">
        Supports classification and regression tasks · Auto-detects target column · No code required
      </div>
    </div>
    """, unsafe_allow_html=True)
