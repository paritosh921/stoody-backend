FROM python:3.12-slim AS base

WORKDIR /app

# Install shared lib first (cached layer)
COPY libs/exampen-common-py /tmp/exampen-common-py
RUN pip install --no-cache-dir /tmp/exampen-common-py && \
    rm -rf /tmp/exampen-common-py

# Install service deps
COPY services/svc-student-bff/pyproject.toml .
RUN pip install --no-cache-dir .

# Copy service source
COPY services/svc-student-bff/src/ src/

EXPOSE 8010

CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8010"]
