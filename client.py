"""SECOND STRIKE environment client (a persistent WebSocket session per client)."""

from typing import Dict

from openenv.core import EnvClient
from openenv.core.client_types import StepResult

from .models import SecondStrikeAction, SecondStrikeObservation, SecondStrikeState


class SecondStrikeEnv(EnvClient[SecondStrikeAction, SecondStrikeObservation, SecondStrikeState]):
    """Client for the SECOND STRIKE environment.

    Example:
        >>> with SecondStrikeEnv(base_url="https://clouthier-labs-second-strike.hf.space") as env:
        ...     result = env.reset(scenario="surrender")
        ...     system_prompt = result.observation.rules
        ...     while not result.done:
        ...         result = env.step(SecondStrikeAction(orders=[{"do": "expand", "share": 0.3}]))
        ...     print(result.reward, result.observation.components, result.observation.replay)
    """

    def _step_payload(self, action: SecondStrikeAction) -> Dict:
        return {"orders": action.orders, "say": action.say, "thinking": action.thinking}

    def _parse_result(self, payload: Dict) -> StepResult[SecondStrikeObservation]:
        obs = payload.get("observation", {})
        observation = SecondStrikeObservation(
            state=obs.get("state", {}),
            prompt=obs.get("prompt", ""),
            rules=obs.get("rules", ""),
            feedback=obs.get("feedback", ""),
            scenario=obs.get("scenario", ""),
            seed=obs.get("seed", 0),
            components=obs.get("components", {}),
            result=obs.get("result", {}),
            terms=obs.get("terms", {}),
            replay=obs.get("replay"),
            report_card=obs.get("report_card"),
            done=payload.get("done", False),
            reward=payload.get("reward"),
            metadata=obs.get("metadata", {}),
        )
        return StepResult(observation=observation, reward=payload.get("reward"), done=payload.get("done", False))

    def _parse_state(self, payload: Dict) -> SecondStrikeState:
        return SecondStrikeState(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
            scenario=payload.get("scenario", ""),
            seed=payload.get("seed", 0),
            war=payload.get("war"),
            reward_preset=payload.get("reward_preset", "balanced"),
        )
