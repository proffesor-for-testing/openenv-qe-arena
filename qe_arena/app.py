import logging
import threading

from openenv.core.env_server.http_server import create_app

from . import sandbox
from .environment import QEAction, QEEnvironment, QEObservation

app = create_app(QEEnvironment, QEAction, QEObservation, env_name="qe_arena", max_concurrent_envs=8)


def _warm_sandbox() -> None:
    # Each sandbox worker compiles python.wasm (~3 s on 2 vCPU). Do it now, off the request path, so
    # /health answers at once and the first episode does not wait.
    try:
        sandbox.warm()
    except Exception:  # noqa: BLE001 - logged here; the next run_calls raises the same error
        logging.getLogger(__name__).exception("sandbox warm-up failed")


threading.Thread(target=_warm_sandbox, name="sandbox-warm", daemon=True).start()
