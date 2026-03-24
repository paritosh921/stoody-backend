FROM python:3.12-slim AS base

WORKDIR /app

# Install shared library first (cache-friendly layer)
COPY libs/exampen-common-py /tmp/exampen-common-py
RUN pip install --no-cache-dir /tmp/exampen-common-py && rm -rf /tmp/exampen-common-py

# Install service dependencies
COPY services/svc-invig-console/pyproject.toml .
RUN pip install --no-cache-dir .

# Copy service source
COPY services/svc-invig-console/src/ src/

EXPOSE 8000

CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]
