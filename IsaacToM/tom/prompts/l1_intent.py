"""Level-1 ToM: "What does my opponent intend to do?" (1st-order belief).

Matches progress.md definition: level-1 = I infer your mental state.
Used by both perspectives — the prompt adapts to self_role.
"""
from __future__ import annotations

SYSTEM_L1 = (
    "You perform first-order Theory-of-Mind reasoning for a driving game. "
    "Looking at the top-down image and the numeric state, infer the OPPONENT's current "
    "intent. Intent = which goal are they trying to reach, or are they stalling/observing. "
    "Base your reasoning on observable motion cues (heading vector, lateral drift, speed, "
    "yaw rate) and geometry. Output strict JSON only."
)

USER_L1_TEMPLATE = (
    "State (JSON):\n{state_json}\n\n"
    "You are the {self_role}. The opponent is the {opp_role}. "
    "Decide which of {{goal_A, goal_B, undecided, observing}} best describes the opponent's "
    "current intent. Provide a confidence in [0,1] and a 1–2 sentence rationale that cites "
    "specific motion cues (e.g. 'heading 112° with lateral drift toward -X')."
)

SCHEMA_L1 = {
    "type": "object",
    "additionalProperties": False,
    "required": ["opponent_intent", "confidence", "rationale", "cues"],
    "properties": {
        "opponent_intent": {"type": "string", "enum": ["goal_A", "goal_B", "undecided", "observing"]},
        "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
        "rationale": {"type": "string"},
        "cues": {
            "type": "array",
            "items": {"type": "string"},
            "minItems": 1,
            "maxItems": 5,
        },
    },
}
