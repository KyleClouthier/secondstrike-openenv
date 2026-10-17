"""FastAPI app for the SECOND STRIKE environment (POST /reset, POST /step, GET /state, GET /schema, WS /ws).

Run locally:  uvicorn server.app:app --host 0.0.0.0 --port 8000
"""

from openenv.core.env_server.http_server import create_app

try:
    from ..models import SecondStrikeAction, SecondStrikeObservation
    from .second_strike_environment import SecondStrikeEnvironment
except ImportError:
    from models import SecondStrikeAction, SecondStrikeObservation
    from server.second_strike_environment import SecondStrikeEnvironment

# The public game server plays 2 wars at once per address, so one host runs 2 sessions at a time.
app = create_app(
    SecondStrikeEnvironment,
    SecondStrikeAction,
    SecondStrikeObservation,
    env_name="second_strike",
    max_concurrent_envs=2,
)


def main(host: str = "0.0.0.0", port: int = 8000):
    import uvicorn

    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    main(port=args.port)
