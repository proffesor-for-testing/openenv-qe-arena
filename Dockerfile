# Builder: the CPython-WASI runtime the sandbox executes. build_wasi_runtime.py downloads
# brettcannon/cpython-wasi-build v3.14.7, checks its sha256, compiles the stdlib inside python.wasm
# (unchecked-hash pycs, sorted order) and packs a byte-reproducible python314.zip; it exits non-zero
# unless both python.wasm and the zip match qe_arena/runtime_pins.json. Base images are pinned by the
# digest of their multi-arch index; wasmtime installs with --require-hashes.
FROM python:3.12-slim@sha256:a0f57064cbdcd6285fa45f51237ab0b35ea8ebaed06a1f5973e89e19dfb86704 AS wasi-runtime
WORKDIR /build
COPY requirements-wasmtime.lock /build/requirements-wasmtime.lock
RUN pip install --no-cache-dir --require-hashes -r /build/requirements-wasmtime.lock
COPY scripts/build_wasi_runtime.py qe_arena/runtime_pins.json /build/
RUN python /build/build_wasi_runtime.py --out /opt/qe-arena/wasi --pins /build/runtime_pins.json

FROM python:3.12-slim@sha256:a0f57064cbdcd6285fa45f51237ab0b35ea8ebaed06a1f5973e89e19dfb86704
LABEL org.opencontainers.image.source="https://github.com/proffesor-for-testing/openenv-qe-arena"
LABEL org.opencontainers.image.description="OpenEnv Arena training environment: QE bug repair and mutation-kill tasks"
LABEL org.opencontainers.image.licenses="MIT"
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends git ca-certificates tini && rm -rf /var/lib/apt/lists/*
COPY requirements.lock requirements-wasmtime.lock /app/
RUN pip install --no-cache-dir --require-hashes -r /app/requirements-wasmtime.lock \
    && pip install --no-cache-dir -r /app/requirements.lock && useradd --uid 10001 --create-home arena
COPY --from=wasi-runtime /opt/qe-arena/wasi /opt/qe-arena/wasi
COPY qe_arena /app/qe_arena
COPY openenv.yaml /app/openenv.yaml
COPY THIRD_PARTY_NOTICES /app/THIRD_PARTY_NOTICES
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 ENABLE_WEB_INTERFACE=false QE_ARENA_WASI_ROOT=/opt/qe-arena/wasi
USER 10001
EXPOSE 8000
ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["python", "-m", "uvicorn", "qe_arena.app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
