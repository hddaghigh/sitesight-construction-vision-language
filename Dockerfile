# CPU inference image: FastAPI + ONNX Runtime, no torch. Model weights are NOT baked in:
# mount an exported model directory (fp32 or int8) and point SITESIGHT_MODEL_DIR at it.
#
#   docker build -t sitesight .
#   docker run --rm -p 8000:8000 -v /path/to/export/int8:/model:ro sitesight
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    SITESIGHT_MODEL_DIR=/model

WORKDIR /app
COPY pyproject.toml README.md ./
COPY sitesight ./sitesight
# runtime dependencies only (pyproject [dependency-groups].serve); needs pip >= 25.1
RUN pip install --upgrade pip && pip install --group serve .

RUN useradd --create-home --uid 10001 app
USER app
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s \
    CMD python -c "import urllib.request as u; u.urlopen('http://127.0.0.1:8000/health', timeout=4)"
CMD ["uvicorn", "--factory", "sitesight.serve.app:build_app", "--host", "0.0.0.0", "--port", "8000"]
