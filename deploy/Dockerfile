# Hugging Face Space (Docker SDK) for the swarm parameter server.
# HF Spaces serve the container on port 7860.
FROM python:3.11-slim

WORKDIR /app

# CPU-only torch keeps the image small (a Space CPU box has no GPU anyway).
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Application code (swarm/ shared package + server/ + worker/ + corpus).
COPY swarm /app/swarm
COPY server /app/server
COPY worker /app/worker

# Aggregation behaviour is configured via Space "Variables" (override as needed).
ENV SWARM_AGG_MODE=sync \
    SWARM_WORLD_SIZE=4 \
    SWARM_CHECKPOINT=/data/global_model.safetensors \
    PORT=7860 \
    PYTHONUNBUFFERED=1

EXPOSE 7860

CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860"]
