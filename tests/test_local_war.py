"""End to end on this machine: a local SECOND STRIKE game server, this OpenEnv server in front of it, and the official
client playing one whole lockstep Hegemon war with a simple rules bot. Checks the observation, the lockstep turns and
the reward at the end. Usage (from this folder): python tests/test_local_war.py
"""

import os
import subprocess
import sys
import tempfile
import time

import httpx

GAME = os.environ.get("SS_GAME_DIR", "C:/clouthier-labs/ideas/games/secondstrike")
GAME_PORT, ENV_PORT = 8795, 8794
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def wait_up(url: str) -> None:
    for _ in range(100):
        try:
            httpx.get(url, timeout=2)
            return
        except httpx.HTTPError:
            time.sleep(0.3)
    raise RuntimeError("not up: " + url)


def main() -> int:
    data = tempfile.mkdtemp(prefix="ss-openenv-")
    genv = dict(os.environ, PORT=str(GAME_PORT), DATA_DIR=data, SS_TEST_AUTH="1", FIRESTORE="", K_SERVICE="",
                DISCORD_BOT_TOKEN="", DISCORD_CLIENT_ID="", TWITCH_CLIENT_ID="", NERUVA_API_KEY="", DEEPSEEK_API_KEY="",
                AGENT_DOOR="1")
    game = subprocess.Popen(["node", "server/server.js"], cwd=GAME, env=genv, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    eenv = dict(os.environ, SECONDSTRIKE_URL=f"http://localhost:{GAME_PORT}", SECONDSTRIKE_MODEL="rules-bot",
                SECONDSTRIKE_REWARD="balanced")
    srv = subprocess.Popen([sys.executable, "-m", "uvicorn", "server.app:app", "--port", str(ENV_PORT)], cwd=HERE, env=eenv,
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    fails = 0

    def check(ok: bool, what: str) -> None:
        nonlocal fails
        print(("ok   " if ok else "FAIL ") + what, flush=True)
        if not ok:
            fails += 1

    try:
        wait_up(f"http://localhost:{GAME_PORT}/health")
        wait_up(f"http://localhost:{ENV_PORT}/health")
        from second_strike import SecondStrikeAction, SecondStrikeEnv

        t0 = time.time()
        with SecondStrikeEnv(base_url=f"http://localhost:{ENV_PORT}").sync() as env:
            r = env.reset(scenario="hegemon", seed=3061001)
            o = r.observation
            check(o.scenario == "hegemon" and o.seed == 3061001, f"reset starts the Hegemon war on its first seed ({o.scenario} {o.seed})")
            check("SECOND STRIKE" in o.rules and "ONE JSON object" in o.rules, f"the rules and reply format come with reset ({len(o.rules)} chars)")
            check(bool(o.state.get("you")) and (o.state.get("lockstep") or {}).get("waiting") is True,
                  "the state is the nation's, and the clock is holding for orders: " + str({k: o.state.get(k) for k in ("status", "lockstep", "note")})[:300])
            ticks, steps = [], 0
            while not r.done and steps < 200:
                you = r.observation.state.get("you") or {}
                orders = [{"do": "expand", "share": 0.3}]
                nb = r.observation.state.get("neighbours") or []
                weak = [n for n in nb if n.get("kind") == "AI nation" and "PACT" not in n.get("relation", "")]
                if weak and you.get("troops", 0) > 0:
                    orders.append({"do": "attack", "target": weak[0]["id"], "share": 0.3})
                r = env.step(SecondStrikeAction(orders=orders, say="" if steps else "Hello from OpenEnv.", thinking="rules bot"))
                steps += 1
                tk = (r.observation.state.get("war") or {}).get("tick")
                if tk is not None:
                    ticks.append(tk)
            mid = ticks[:-1]  # the last step moves the war into its end, not to another turn
            gaps = {b - a for a, b in zip(mid, mid[1:])}
            check(gaps == {120}, f"every step moves the war exactly 120 ticks, 12 s of war (gaps seen: {sorted(gaps)})")
            check(r.done, f"the war ends within {steps} steps")
            comps = r.observation.components or {}
            check(set(comps) >= {"land", "won", "survived", "no_midnight", "pacts_kept", "kept_terms"}, f"the last step carries the reward parts: {comps}")
            check(isinstance(r.reward, (int, float)) and 0.0 <= r.reward <= 1.0, f"the reward is the weighted score: {r.reward}")
            check(bool(r.observation.replay) and "ref=openenv" in r.observation.replay, f"the replay link comes back: {r.observation.replay}")
            print(f"whole war: {steps} steps in {time.time() - t0:.0f} s", flush=True)
    except Exception as e:  # noqa: BLE001
        check(False, f"crashed: {type(e).__name__}: {e}")
    finally:
        srv.terminate()
        game.terminate()
    print("all passed" if not fails else f"{fails} failed")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
