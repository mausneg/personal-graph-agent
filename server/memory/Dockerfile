FROM python:3.10-slim

WORKDIR /app

COPY server/requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY server/memory/memory_server.py memory_server.py

EXPOSE 8003

CMD ["python3", "memory_server.py"]