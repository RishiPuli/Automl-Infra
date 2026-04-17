FROM python:3.11-slim

# ── Build metadata ─────────────────────────────────────────────────────────────
LABEL maintainer="CloudAutoML"
LABEL description="Production-grade AutoML platform"

# ── System dependencies ───────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# ── Working directory ─────────────────────────────────────────────────────────
WORKDIR /app

# ── Install Python dependencies (layer-cached) ────────────────────────────────
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# ── Copy source ───────────────────────────────────────────────────────────────
COPY . .

# ── Create runtime directories ────────────────────────────────────────────────
RUN mkdir -p data models reports logs mlruns

# ── Streamlit / MLflow ports ──────────────────────────────────────────────────
EXPOSE 8501
EXPOSE 5000

# ── Health check ─────────────────────────────────────────────────────────────
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
  CMD curl -f http://localhost:8501/_stcore/health || exit 1

# ── Default command ───────────────────────────────────────────────────────────
CMD ["streamlit", "run", "app/streamlit_app.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0", \
     "--server.headless=true", \
     "--browser.gatherUsageStats=false"]
