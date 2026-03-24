FROM python:3.12-slim AS base

WORKDIR /app

# Install shared library first (cache layer)
COPY libs/exampen-common-py /libs/exampen-common-py
RUN pip install --no-cache-dir /libs/exampen-common-py

# Install service dependencies
COPY services/svc-copy-upload/pyproject.toml .
RUN pip install --no-cache-dir .

# Copy service source
COPY services/svc-copy-upload/src ./src
COPY services/svc-copy-upload/migrations ./migrations

ENV SERVICE_NAME=svc-copy-upload
EXPOSE 8006

CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8006"]
