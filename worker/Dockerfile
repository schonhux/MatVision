# Build context is the REPO ROOT (see infra/docker/docker-compose.yml: `context: ..`,
# `dockerfile: worker/Dockerfile`), because this image needs files from BOTH api/app
# (shared DB models/config/storage — single source of truth, see dev/DECISIONS.md
# ADR-009) and worker/app (the actual stage/task code). Two COPY instructions into
# the same destination directory merge their contents rather than overwriting.
FROM python:3.11-slim

WORKDIR /code

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

COPY worker/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Shared DB models/config/storage from the API package...
COPY api/app ./app
# ...merged with worker-specific stage/task code (adds app/stages/, app/vision/,
# app/tasks.py; does not remove anything the previous COPY placed).
COPY worker/app ./app

# Layer 3+ stages import the torch-free feature math from ml/ (wrestling_features,
# identity, splits). Copied whole so the same code runs in the worker and in tests.
COPY ml ./ml

# api/app also contains routers/, main.py, security.py, schemas.py, deps.py,
# queue.py — none of which the worker needs or should import. Removing them keeps
# the image lean and makes it obvious (via ImportError) if worker code accidentally
# tries to depend on API-only modules.
RUN rm -rf ./app/routers ./app/main.py ./app/security.py ./app/schemas.py \
           ./app/deps.py ./app/queue.py

CMD ["dramatiq", "app.tasks", "--processes", "1", "--threads", "4"]
