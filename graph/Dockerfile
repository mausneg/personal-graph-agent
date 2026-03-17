FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl ca-certificates gnupg \
    && curl -fsSL https://deb.nodesource.com/setup_20.x | bash - \
    && apt-get install -y --no-install-recommends nodejs \
    && rm -rf /var/lib/apt/lists/*

COPY graph/requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY graph/ graph/

EXPOSE 8002

CMD ["uvicorn", "graph.app:app", "--host", "0.0.0.0", "--port", "8002"]