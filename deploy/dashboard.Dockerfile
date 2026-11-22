# SolarSentinel dashboard (C10) and replay driver (C7b + C9 tracker/monitor): no torch (plan 6.6).
# Data are fetched at first start into the data volume over public HTTPS, sha256-checked (solarsentinel.serve.fetch).
FROM python:3.13.5-slim-bookworm
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1 PYTHONPATH=/app/src
WORKDIR /app
COPY deploy/requirements-dashboard.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt \
 && useradd --uid 10001 --create-home --shell /usr/sbin/nologin app \
 && mkdir -p /app/data && chown app /app/data
COPY configs/ configs/
COPY src/ src/
COPY deploy/dashboard/ deploy/dashboard/
COPY .streamlit/ .streamlit/
USER app
EXPOSE 8501 9101
HEALTHCHECK --interval=15s --timeout=3s --start-period=60s --retries=5 \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:' + __import__('os').environ.get('PORT','8501') + '/healthz', timeout=2).status == 200 else 1)"
CMD ["sh", "-c", "exec streamlit run src/solarsentinel/serve/dashboard.py --server.address 0.0.0.0 --server.port ${PORT:-8501}"]
