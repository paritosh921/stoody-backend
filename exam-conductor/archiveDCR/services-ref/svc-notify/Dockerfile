FROM python:3.12-slim AS base

WORKDIR /app

COPY pyproject.toml .
RUN pip install --no-cache-dir .

COPY src/ src/

EXPOSE 8007

CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8007"]
