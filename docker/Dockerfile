# ── Dockerfile.dev ──────────────────────────────────────
# Development image for Craster HR Policy RAG chatbot
#
# Hot reload + dev tools + debug logging
#
# Build: docker build -f docker/Dockerfile.dev -t craster-rag:dev .
# Run:   docker compose -f docker/docker-compose.yml up

FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src \
    ENVIRONMENT=local \
    DEBUG=true \
    LOG_LEVEL=DEBUG

WORKDIR /app

# system deps
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    && rm -rf /var/lib/apt/lists/*

# install all deps including dev tools
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# copy source — overridden by volume mount in docker-compose
# so code changes reflect immediately without rebuild
COPY src/ ./src/
COPY config.py .
COPY pyproject.toml .
COPY frontend/ ./frontend/

RUN pip install --no-cache-dir -e .

# create directories
RUN mkdir -p data/documents/hr_policy logs mlflow/experiments

EXPOSE 8000

# development server with hot reload
# watches for file changes and restarts automatically
CMD ["uvicorn", "craster_rag.api.app:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--reload", \
     "--reload-dir", "/app/src"]
