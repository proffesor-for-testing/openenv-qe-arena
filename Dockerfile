FROM python:3.12-slim
LABEL org.opencontainers.image.source="https://github.com/proffesor-for-testing/openenv-qe-arena"
LABEL org.opencontainers.image.description="OpenEnv Arena training environment: QE bug repair and mutation-kill tasks"
LABEL org.opencontainers.image.licenses="MIT"
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends git ca-certificates tini && rm -rf /var/lib/apt/lists/*
COPY requirements.lock /app/requirements.lock
RUN pip install --no-cache-dir -r /app/requirements.lock && useradd --uid 10001 --create-home arena
COPY qe_arena /app/qe_arena
COPY openenv.yaml /app/openenv.yaml
COPY THIRD_PARTY_NOTICES /app/THIRD_PARTY_NOTICES
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 ENABLE_WEB_INTERFACE=false
USER 10001
EXPOSE 8000
ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["python", "-m", "uvicorn", "qe_arena.app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
