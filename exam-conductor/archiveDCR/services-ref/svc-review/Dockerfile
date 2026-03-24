FROM python:3.12-slim AS base

WORKDIR /app

# Install system deps for asyncpg
RUN apt-get update && \
    apt-get install -y --no-install-recommends gcc libpq-dev && \
    rm -rf /var/lib/apt/lists/*

# Install shared lib first (cached layer)
COPY libs/exampen-common-py /tmp/exampen-common-py
RUN pip install --no-cache-dir /tmp/exampen-common-py && \
    rm -rf /tmp/exampen-common-py

# Install service deps
COPY services/svc-review/pyproject.toml .
RUN pip install --no-cache-dir .

# Copy service source
COPY services/svc-review/src/ src/
COPY services/svc-review/migrations/ migrations/

EXPOSE 8007

CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8007"]
