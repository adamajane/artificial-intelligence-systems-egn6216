# SolarSentinel scorer (C7a): the frozen bundle behind FastAPI, CPU only (plan 6.6).
# Build context: repo root. The bundle is fetched at build time over public HTTPS and checked against the bundle ID
# pinned in configs/experiments.yaml, so the image is self-contained (Cloud Run) and holds no credentials.
FROM python:3.13.5-slim-bookworm
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1 PYTHONPATH=/app/src
WORKDIR /app
COPY deploy/requirements-scorer.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt
COPY configs/ configs/
COPY src/ src/
RUN python -m solarsentinel.serve.fetch bundle \
 && useradd --uid 10001 --no-create-home --shell /usr/sbin/nologin app
USER app
ENV TORCH_NUM_THREADS=2 SOLARSENTINEL_MODE=probability
EXPOSE 8000
HEALTHCHECK --interval=10s --timeout=3s --start-period=40s --retries=3 \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:' + __import__('os').environ.get('PORT','8000') + '/healthz', timeout=2).status == 200 else 1)"
CMD ["sh", "-c", "exec uvicorn solarsentinel.serve.api:app --host 0.0.0.0 --port ${PORT:-8000} --no-access-log"]
