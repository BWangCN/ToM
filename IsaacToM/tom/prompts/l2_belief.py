"""Level-2 ToM: "What does my opponent believe about MY intent?" (2nd-order).

progress.md: level-2 = I infer your inference about MY mental state. For the
evader, this is the substrate of a feint. For the defender, this is "am I being
baited right now".
"""
from __future__ import annotations

SYSTEM_L2 = (
    "You perform second-order Theory-of-Mind reasoning. You already have a level-1 estimate "
    "of the opponent's intent. Now estimate the opponent's belief about YOUR intent: which "
    "goal does the opponent currently think YOU are heading to? Consider what your motion "
    "looks like from the opponent's viewpoint — heading, lateral drift, recent shaping. "
    "Output strict JSON only."
)

USER_L2_TEMPLATE = (
    "State (JSON):\n{state_json}\n\n"
    "Level-1 output (your estimate of opponent's intent):\n{l1_json}\n\n"
    "You are the {self_role}. From the OPPONENT's perspective, estimate the probability that "
    "the opponent currently believes you are heading to goal_A vs goal_B vs undecided. "
    "Also rate the `commitment_evidence` — how strongly your own recent motion commits you "
    "to one reading (0 = fully ambiguous, 1 = unmistakably toward one goal). If you are the "
    "evader, a high commitment_evidence toward the fake goal is what makes a feint work."
)

SCHEMA_L2 = {
    "type": "object",
    "additionalProperties": False,
    "required": ["opponent_belief", "commitment_evidence", "rationale"],
    "properties": {
        "opponent_belief": {
            "type": "object",
            "additionalProperties": False,
            "required": ["p_goal_A", "p_goal_B", "p_undecided"],
            "properties": {
                "p_goal_A":     {"type": "number", "minimum": 0.0, "maximum": 1.0},
                "p_goal_B":     {"type": "number", "minimum": 0.0, "maximum": 1.0},
                "p_undecided":  {"type": "number", "minimum": 0.0, "maximum": 1.0},
            },
        },
        "commitment_evidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
        "rationale": {"type": "string"},
    },
}
