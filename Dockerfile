# Reproducible build: exact pins from requirements.lock, non-root runtime,
# health check, seeded demo data baked in at build time (regenerated, not copied).
ARG BASE_IMAGE=python:3.12-slim
FROM ${BASE_IMAGE}

WORKDIR /app

# dependencies first for layer caching
COPY requirements.lock .
RUN pip install --no-cache-dir -r requirements.lock

# application code + policy corpus + data generator (no raw data copied in)
COPY src/ src/
COPY policy/ policy/
COPY data/generate_data.py data/generate_data.py
COPY app.py run_all.py pyproject.toml ./

# bake the seeded demo datasets into the image (deterministic: seeds 11 / 4242)
RUN python data/generate_data.py --set both

# non-root runtime user owning only what it needs to write
RUN useradd --create-home copilot && chown -R copilot:copilot /app
USER copilot

EXPOSE 8000
ENV PYTHONUNBUFFERED=1

# liveness probe without curl (slim image): stdlib urllib against /health
HEALTHCHECK --interval=30s --timeout=3s --start-period=15s --retries=3 \
  CMD python -c "import urllib.request as u; u.urlopen('http://127.0.0.1:8000/health', timeout=2)" || exit 1

# deterministic mode by default; set LLM_PROVIDER / EMBEDDINGS to enable LLM paths
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
