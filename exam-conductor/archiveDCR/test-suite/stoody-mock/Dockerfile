FROM python:3.12-slim

WORKDIR /app

RUN pip install --no-cache-dir \
    fastapi>=0.111,<1 \
    uvicorn[standard]>=0.29,<1 \
    pyjwt[crypto]>=2.8,<3 \
    cryptography>=42,<44

COPY main.py data.py keys.py ./

EXPOSE 9100

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "9100"]
