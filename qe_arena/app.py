from openenv.core.env_server.http_server import create_app

from .environment import QEAction, QEEnvironment, QEObservation

app = create_app(QEEnvironment, QEAction, QEObservation, env_name="qe_arena", max_concurrent_envs=8)
