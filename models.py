"""Data models for the SECOND STRIKE environment.

One step is one turn of a war: the model's orders for its nation. The observation is the nation's state as the game
server sends it, plus a ready-made prompt, and at the end of the war the reward and its parts.
"""

from typing import Any, Dict, List, Optional

from openenv.core.env_server.types import Action, Observation, State
from pydantic import Field


class SecondStrikeAction(Action):
    """One turn of orders. Every order is listed in the rules (observation.rules on reset), for example
    {"do": "attack", "target": 12, "share": 0.4}, {"do": "ally", "target": "POLAND"}, {"do": "build", "what": "silo"},
    {"do": "nuke", "target": 17}, and in the Surrender war {"do": "surrender"} or {"do": "refuse"}."""

    orders: List[Dict[str, Any]] = Field(default_factory=list, description="Up to 8 orders for this turn")
    say: str = Field(default="", description="Optional public message to every nation in the war")
    thinking: str = Field(default="", description="Optional private reasoning, kept in the war record")


class SecondStrikeObservation(Observation):
    """The nation's state this turn. `prompt` is a ready-to-use user message for an LLM; `rules` (on reset) is the
    system prompt. When `done`, `reward` is the war's score and `metadata["components"]` its parts."""

    state: Dict[str, Any] = Field(default_factory=dict, description="This turn's state from the game server")
    prompt: str = Field(default="", description="A ready-to-use user message: feedback, then the state as JSON")
    rules: str = Field(default="", description="The full rules and reply format (sent on reset)")
    feedback: str = Field(default="", description="Orders the game refused last turn, if any")
    scenario: str = Field(default="", description="The Midnight Index scenario of this war")
    seed: int = Field(default=0, description="The war's fixed seed")
    components: Dict[str, float] = Field(default_factory=dict, description="At the end: every part of the reward (land, won, survived, no_midnight, pacts_kept, promises_kept, not_two_faced, kept_terms, ...)")
    result: Dict[str, Any] = Field(default_factory=dict, description="At the end: the server's record of this nation (place, land, nukes, betrayals, struckMidnight, ...)")
    terms: Dict[str, bool] = Field(default_factory=dict, description="Surrender war: whether terms were offered, accepted, broken or refused")
    replay: Optional[str] = Field(default=None, description="Link to watch this war move by move")
    report_card: Optional[str] = Field(default=None, description="This agent's public record")


class SecondStrikeState(State):
    """Episode bookkeeping."""

    scenario: str = Field(default="")
    seed: int = Field(default=0)
    war: Optional[str] = Field(default=None)
    reward_preset: str = Field(default="balanced")
