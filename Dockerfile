FROM python:3.11-slim

WORKDIR /app

# Install system deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    git curl \
    && rm -rf /var/lib/apt/lists/*

# Install uv for fast dependency management
RUN curl -LsSf https://astral.sh/uv/install.sh | sh && \
    mv /root/.local/bin/uv /usr/local/bin/uv && \
    mv /root/.local/bin/uvx /usr/local/bin/uvx

# Copy project files (v2 - with coding+toolcalling tasks)
COPY pyproject.toml /app/
COPY openenv.yaml /app/
COPY dataqa_env/ /app/dataqa_env/
COPY inference.py /app/
COPY README.md /app/

# Install dependencies
RUN uv sync --no-editable 2>/dev/null || pip install -e .

# Set environment
ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONPATH="/app:$PYTHONPATH"

# Health check — HF Spaces uses port 8000
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

EXPOSE 8000

ENV ENABLE_WEB_INTERFACE=true
CMD ["uvicorn", "dataqa_env.server.app:app", "--host", "0.0.0.0", "--port", "8000"]
