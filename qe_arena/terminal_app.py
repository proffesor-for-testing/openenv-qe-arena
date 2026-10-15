"""The terminal-interface server (Dockerfile.terminal). The JSON-interface server is app.py."""

import logging
import threading

from openenv.core.env_server.http_server import create_app

from . import sandbox, shell
from .terminal_env import TerminalAction, TerminalEnvironment, TerminalObservation

app = create_app(TerminalEnvironment, TerminalAction, TerminalObservation, env_name="qe_arena_terminal",
                 max_concurrent_envs=8)


def _warm() -> None:
    try:
        shell.mode()  # logs the isolation mode once, at start-up
        sandbox.warm()
    except Exception:  # noqa: BLE001 - logged here; the first episode raises the same error
        logging.getLogger(__name__).exception("warm-up failed")


threading.Thread(target=_warm, name="sandbox-warm", daemon=True).start()
