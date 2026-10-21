# API image: two-stage, non-root, health-checked.
FROM python:3.11-slim AS build

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Runtime set comes straight from the pinned lock (dev tooling filtered out).
COPY requirements.lock ./requirements.lock
RUN grep -vE '^(pytest==|pytest-cov==|coverage==|ruff==|httpx==)' requirements.lock > runtime.lock \
    && pip install --no-cache-dir -r runtime.lock

# Install the application itself with no re-resolution.
COPY pyproject.toml README.md ./
COPY src ./src
COPY scripts ./scripts
RUN pip install --no-cache-dir --no-deps .

FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN useradd --create-home --shell /usr/sbin/nologin appuser

WORKDIR /app

# Pre-create the mounted data dir with appuser ownership before the volume.
RUN mkdir -p /app/data && chown -R appuser /app

COPY --from=build /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=build /usr/local/bin /usr/local/bin

USER appuser

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4)" || exit 1

EXPOSE 8000

CMD ["uvicorn", "supply_twin.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
